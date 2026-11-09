// bk_eval.c - headless evaluation harness for PufferLib 5.0 Breakout.
//
// The simulator and the policy network are PufferLib's own, compiled in
// unmodified:
//   src/puffercpu.c            PufferNet inference (encoder -> MinGRU -> decoder)
//   ocean/breakout/breakout.h  the Breakout environment
// This harness adds per-episode seeding and logging, a step cap and optional
// perturbation protocols (see bk_common.h).
//
// Network policies choose actions through forward_puffernet(), PufferLib's
// stock code path: softmax *sampling* from the decoder logits. --argmax
// switches to greedy decoding.
//
// Build: harness/build.sh

#define _GNU_SOURCE
#include "src/puffercpu.c"
#include "ocean/breakout/breakout.h"
#include "bk_common.h"

typedef struct {
    int32_t episode;
    int32_t step;
    int32_t tick;
    int32_t action_policy;
    int32_t action_applied;
    float logits[4];
    float obs[OBS_SIZE];
    float paddle_x, ball_x, ball_y, ball_vx, ball_vy, paddle_width;
    int32_t score, num_balls;
} TraceRecord;

static void usage(void) {
    fprintf(stderr,
        "usage: bk_eval --puffer-root DIR --policy net|random|noop [--weights F]\n"
        "    [--episodes N] [--seed-base S] [--max-steps M] [--sticky P]\n"
        "    [--random-start-max K] [--argmax] [--out CSV] [--label NAME]\n"
        "    [--trace FILE] [--trace-episodes N] [-- --env.key=value ...]\n");
    exit(2);
}

int main(int argc, char** argv) {
    const char* puffer_root = NULL;
    const char* weights_path = NULL;
    const char* out_path = NULL;
    const char* trace_path = NULL;
    const char* label = "run";
    int policy = -1;
    int episodes = 10;
    long seed_base = 1;
    int trace_episodes = 1;
    char* ini_argv[64];
    int ini_argc = 0;
    static Runner r;
    r.max_steps = 30000;

    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (strcmp(a, "--") == 0) {
            for (int j = i + 1; j < argc && ini_argc < 64; j++) {
                ini_argv[ini_argc++] = argv[j];
            }
            break;
        }
        if (strcmp(a, "--argmax") == 0) {
            r.argmax = 1;
            continue;
        }
        if (i + 1 >= argc) usage();
        if (strcmp(a, "--puffer-root") == 0) puffer_root = argv[++i];
        else if (strcmp(a, "--weights") == 0) weights_path = argv[++i];
        else if (strcmp(a, "--out") == 0) out_path = argv[++i];
        else if (strcmp(a, "--trace") == 0) trace_path = argv[++i];
        else if (strcmp(a, "--label") == 0) label = argv[++i];
        else if (strcmp(a, "--episodes") == 0) episodes = atoi(argv[++i]);
        else if (strcmp(a, "--seed-base") == 0) seed_base = atol(argv[++i]);
        else if (strcmp(a, "--max-steps") == 0) r.max_steps = atol(argv[++i]);
        else if (strcmp(a, "--sticky") == 0) r.sticky = atof(argv[++i]);
        else if (strcmp(a, "--random-start-max") == 0) r.random_start_max = atoi(argv[++i]);
        else if (strcmp(a, "--trace-episodes") == 0) trace_episodes = atoi(argv[++i]);
        else if (strcmp(a, "--policy") == 0) {
            const char* p = argv[++i];
            if (strcmp(p, "net") == 0) policy = POLICY_NET;
            else if (strcmp(p, "random") == 0) policy = POLICY_RANDOM;
            else if (strcmp(p, "noop") == 0) policy = POLICY_NOOP;
            else usage();
        } else usage();
    }
    if (!puffer_root || policy < 0) usage();
    if (policy == POLICY_NET && !weights_path) usage();

    // Opened before runner_init() changes directory.
    FILE* out = out_path ? fopen(out_path, "w") : stdout;
    if (!out) { perror(out_path); return 1; }
    FILE* trace = trace_path ? fopen(trace_path, "wb") : NULL;
    if (trace_path && !trace) { perror(trace_path); return 1; }

    runner_init(&r, puffer_root, policy, weights_path, ini_argc, ini_argv, 0);
    runner_describe(&r, stderr, "BK_EVAL", label);
    fprintf(stderr, "BK_EVAL episodes=%d seed_base=%ld\n", episodes, seed_base);
    runner_csv_header(out);

    for (int ep = 0; ep < episodes; ep++) {
        runner_begin_episode(&r, seed_base + ep);
        while (!r.done) {
            runner_decide(&r);
            if (trace && ep < trace_episodes) {
                TraceRecord t = {0};
                t.episode = ep;
                t.step = (int32_t)r.steps;
                t.tick = r.env.tick;
                t.action_policy = r.a_policy;
                t.action_applied = r.a_applied;
                memcpy(t.logits, r.logits, sizeof(t.logits));
                for (int i = 0; i < OBS_SIZE; i++) t.obs[i] = r.observations[i];
                t.paddle_x = r.env.paddle_x;
                t.ball_x = r.env.ball_x;
                t.ball_y = r.env.ball_y;
                t.ball_vx = r.env.ball_vx;
                t.ball_vy = r.env.ball_vy;
                t.paddle_width = r.env.paddle_width;
                t.score = r.env.score;
                t.num_balls = r.env.num_balls;
                fwrite(&t, sizeof(t), 1, trace);
            }
            runner_step(&r);
        }
        runner_csv_row(&r, out, label, ep);
        fflush(out);
    }

    if (trace) fclose(trace);
    if (out != stdout) fclose(out);
    puf_close(&r.env);
    puf_ini_free(&r.ini);
    return 0;
}
