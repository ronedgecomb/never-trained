// bk_common.h - shared episode runner for the Breakout harnesses.
//
// Included after PufferLib's src/puffercpu.c (PufferNet inference) and
// ocean/breakout/breakout.h (the environment). Nothing here changes either:
// it seeds episodes, calls PufferLib's forward pass, optionally perturbs the
// chosen action (stress protocols) and keeps per-episode statistics.

#pragma once

#include <stdint.h>
#include <unistd.h>

enum { POLICY_NET, POLICY_RANDOM, POLICY_NOOP };

static uint64_t sm64(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double sm_uniform(uint64_t* s) {
    return (double)(sm64(s) >> 11) * (1.0 / 9007199254740992.0);
}

static int sm_int(uint64_t* s, int n) {
    return (int)(sm_uniform(s) * n);
}

typedef struct {
    // Configuration
    int policy;
    int argmax;
    double sticky;
    int random_start_max;
    long max_steps;
    int hidden_size;
    int num_layers;
    int file_floats;
    int need_floats;
    int hold_frames;        // frames each action lasts (the configured frameskip)

    // PufferLib objects
    Ini ini;
    Env env;
    PufferNet* net;
    obs_t observations[OBS_SIZE];
    float actions[NUM_ATNS];
    float rewards[1];
    float terminals[1];
    unsigned char masks[3];

    // Per-episode state
    long seed;
    uint64_t rng_policy;
    uint64_t rng_perturb;
    int random_start_k;
    long steps;
    int prev_action;
    int balls_lost;
    int paddle_hits;
    int prev_hits;
    int sticky_repeats;
    int truncated;
    int done;
    int score;
    long frames;
    char serves[64];
    int n_serves;
    uint32_t action_hash;

    // Last decision
    int a_policy;
    int a_applied;
    float logits[4];
} Runner;

static const char* runner_policy_name(const Runner* r) {
    return r->policy == POLICY_NET ? "net"
        : r->policy == POLICY_RANDOM ? "random" : "noop";
}

// Loads weights (before chdir, so relative paths behave as typed), then enters
// the PufferLib checkout, where config/ and resources/ are resolved.
//
// single_frame = 0: the env applies its configured frameskip inside puf_step()
// (the training-time semantics; used for evaluation).
// single_frame = 1: the env advances one frame per puf_step() and the caller
// holds each action for r->hold_frames frames, so every frame can be drawn.
// The dynamics are the same: puf_step() with frameskip N is N single frames
// with the same action.
static void runner_init(Runner* r, const char* puffer_root, int policy,
        const char* weights_path, int ini_argc, char** ini_argv,
        int single_frame) {
    r->policy = policy;
    Weights* weights = NULL;
    if (policy == POLICY_NET) {
        weights = load_weights(weights_path);
        if (!weights) {
            perror(weights_path);
            exit(1);
        }
    }
    if (chdir(puffer_root) != 0) {
        perror(puffer_root);
        exit(1);
    }
    puf_ini_load_env(&r->ini, "breakout", ini_argc, ini_argv);
    r->hidden_size = puf_ini_get(&r->ini, "policy", "hidden_size");
    r->num_layers = puf_ini_get(&r->ini, "policy", "num_layers");

    Dict* env_sec = puf_ini_section(&r->ini, "env", 0);
    r->hold_frames = dict_get(env_sec, "frameskip");
    if (single_frame) {
        dict_set(env_sec, "frameskip", 1);
    }
    puf_init(&r->env, env_sec);
    memset(r->masks, 1, sizeof(r->masks));
    r->env.agents[0].observations = r->observations;
    r->env.agents[0].actions = r->actions;
    r->env.agents[0].rewards = r->rewards;
    r->env.agents[0].terminals = r->terminals;
    r->env.agents[0].action_mask = r->masks;

    if (policy == POLICY_NET) {
        int act_sizes[] = ACT_SIZES;
        int num_actions = sizeof(act_sizes) / sizeof(act_sizes[0]);
        // Same size check as PufferLib's stock eval main.
        int n = (r->hidden_size * OBS_SIZE + 7) & ~7;
        n = (n + 4 * r->hidden_size + 7) & ~7;
        for (int l = 0; l < r->num_layers; l++) {
            n = (n + 3 * r->hidden_size * r->hidden_size + 7) & ~7;
        }
        r->need_floats = n;
        r->file_floats = weights->size - 7;
        if (!(n - r->file_floats <= 7 && r->file_floats <= n)) {
            fprintf(stderr, "weight file has %d floats, architecture needs %d\n",
                r->file_floats, n);
            exit(1);
        }
        r->net = make_puffernet(weights, 1, OBS_SIZE, r->hidden_size,
            r->num_layers, act_sizes, num_actions);
    }
}

static void runner_describe(const Runner* r, FILE* fp, const char* tool,
        const char* label) {
    const Env* e = &r->env;
    fprintf(fp,
        "%s label=%s policy=%s hidden=%d layers=%d file_floats=%d need=%d "
        "frameskip=%d width=%d height=%d paddle_w=%g ball=%dx%d bricks=%dx%d "
        "speed=%g..%g paddle_speed=%g max_score=%d max_steps=%ld sticky=%g "
        "random_start_max=%d argmax=%d\n",
        tool, label, runner_policy_name(r), r->hidden_size, r->num_layers,
        r->file_floats, r->need_floats, r->hold_frames, e->width, e->height,
        e->initial_paddle_width, e->ball_width, e->ball_height, e->brick_rows,
        e->brick_cols, e->initial_ball_speed, e->max_ball_speed,
        e->paddle_speed, e->max_score, r->max_steps, r->sticky,
        r->random_start_max, r->argmax);
}

// Fresh episode. The env's only randomness is the serve direction, drawn from
// env.rng with rand_r(); PufferLib's action sampling uses libc rand().
static void runner_begin_episode(Runner* r, long seed) {
    r->seed = seed;
    r->rng_policy = 0x1234567ULL ^ ((uint64_t)seed * 0x9E3779B97F4A7C15ULL);
    r->rng_perturb = 0xABCDEF01ULL ^ ((uint64_t)seed * 0xD1B54A32D192ED03ULL);
    r->env.rng = (unsigned int)seed;
    memset(&r->env.log, 0, sizeof(r->env.log));
    puf_reset(&r->env);
    r->actions[0] = 0;
    r->rewards[0] = 0;
    r->terminals[0] = 1;   // makes forward_puffernet zero the recurrent state
    srand((unsigned int)seed);

    r->random_start_k = r->random_start_max > 0
        ? sm_int(&r->rng_perturb, r->random_start_max + 1) : 0;
    r->steps = 0;
    r->prev_action = 0;
    r->balls_lost = 0;
    r->paddle_hits = 0;
    r->prev_hits = 0;
    r->sticky_repeats = 0;
    r->truncated = 0;
    r->done = 0;
    r->score = 0;
    r->frames = 0;
    r->n_serves = 0;
    r->serves[0] = 0;
    r->action_hash = 2166136261u;
}

// Chooses the action for the next agent step from the current observation.
static void runner_decide(Runner* r) {
    int a = 0;
    memset(r->logits, 0, sizeof(r->logits));
    if (r->policy == POLICY_NET) {
        forward_puffernet(r->net, (float*)r->observations, r->actions,
            r->masks, r->terminals);
        a = (int)r->actions[0];
        memcpy(r->logits, r->net->decoder->output, sizeof(r->logits));
        if (r->argmax) {
            a = 0;
            for (int i = 1; i < 3; i++) {
                if (r->logits[i] > r->logits[a]) a = i;
            }
        }
    } else if (r->policy == POLICY_RANDOM) {
        a = sm_int(&r->rng_policy, 3);
    }
    r->a_policy = a;

    // Stress protocols: uniformly random actions for the first K steps, and
    // step-level sticky actions (repeat the previously applied action).
    if (r->steps < r->random_start_k) {
        a = sm_int(&r->rng_perturb, 3);
    } else if (r->sticky > 0.0 && r->steps > 0
            && sm_uniform(&r->rng_perturb) < r->sticky) {
        if (a != r->prev_action) r->sticky_repeats++;
        a = r->prev_action;
    }
    r->a_applied = a;
    r->prev_action = a;
    r->action_hash = (r->action_hash ^ (uint32_t)a) * 16777619u;
}

typedef struct {
    int fired;
    int balls;
} StepMark;

static StepMark runner_mark(const Runner* r) {
    return (StepMark){r->env.balls_fired, r->env.num_balls};
}

// Bookkeeping after env frames have been advanced. `mark` is the state before
// them. Sets r->done when the episode finished (the env has then auto-reset).
static void runner_account(Runner* r, StepMark mark) {
    Env* e = &r->env;
    if (e->log.n >= 1.0f) {
        r->score = (int)e->log.score;
        r->frames = (long)e->log.episode_length;
        if (r->score < e->max_score) r->balls_lost++;
        r->done = 1;
        return;
    }
    if (e->num_balls < mark.balls) {
        r->balls_lost += mark.balls - e->num_balls;
    }
    int relaunched = (mark.fired == 0 && e->balls_fired == 1)
        || (e->num_balls < mark.balls && e->balls_fired == 1);
    if (relaunched && r->n_serves < 63) {
        r->serves[r->n_serves++] = e->ball_vx < 0 ? 'L' : 'R';
        r->serves[r->n_serves] = 0;
    }
    if (e->hits > r->prev_hits) r->paddle_hits += e->hits - r->prev_hits;
    r->prev_hits = e->hits;
}

// One agent step with the env's own frameskip (the training-time semantics).
static void runner_step(Runner* r) {
    StepMark mark = runner_mark(r);
    r->actions[0] = r->a_applied;
    puf_step(&r->env);
    r->steps++;
    runner_account(r, mark);
    if (!r->done && r->steps >= r->max_steps) {
        r->truncated = 1;
        r->done = 1;
        r->score = r->env.score;
        r->frames = r->env.tick;
    }
}

static void runner_csv_header(FILE* fp) {
    fprintf(fp, "label,episode,seed,score,perf,cleared,truncated,frames,agent_steps,"
        "balls_lost,paddle_hits,serves,action_hash,sticky_repeats,random_start_k\n");
}

static void runner_csv_row(const Runner* r, FILE* fp, const char* label, int ep) {
    fprintf(fp, "%s,%d,%ld,%d,%.6f,%d,%d,%ld,%ld,%d,%d,%s,%08x,%d,%d\n",
        label, ep, r->seed, r->score, r->score / (float)r->env.max_score,
        r->score == r->env.max_score, r->truncated, r->frames, r->steps,
        r->balls_lost, r->paddle_hits, r->serves, r->action_hash,
        r->sticky_repeats, r->random_start_k);
}
