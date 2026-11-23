"""Headless NEAT training for the Flappy Bird task in flappy.py.

flappy.py needs pygame, a display and the image files in imgs/ (which are not
in this repository). This script runs the SAME training problem without any of
that, so it works on a server or in CI:

  * same config.txt (population size, activation, mutation rates, ...)
  * same bird physics, pipe gap, pipe speed, spawn rule and fitness rules
  * same network inputs: (bird.y, |bird.y - gap_top|, |bird.y - gap_bottom|)
  * same jump rule: output[0] > 0.5

Differences from flappy.py, all forced by having no images:
  * Sprite sizes are the standard ones for this tutorial's art (bird 34x24,
    pipe 52x320, background 288x512, all doubled by scale2x). Those files are
    missing from the repo, so the sizes are an assumption, not a measurement.
  * Collision is a rectangle test (bird 68x48 against the 104-wide pipes). The
    original uses pixel masks, which ignore the transparent corners of the
    bird sprite, so it is very slightly more forgiving.
  * A generation also ends as soon as any bird reaches fitness_threshold,
    because training would stop right after that generation anyway. The
    original would keep playing until every bird had died.

Usage:
    python headless_train.py                      # one run, seed 0, 20 generations
    python headless_train.py --seed 7 --generations 50
    python headless_train.py --runs 30 --json results.json
"""
import argparse
import json
import math
import os
import random
import statistics

import neat

# ---- constants copied from flappy.py (see the comments there) --------------
WIN_WIDTH = 476          # BG_IMG.get_width() - 100, assuming a 288 px wide bg.png
BIRD_X, BIRD_Y0 = 200, 310
BIRD_W, BIRD_H = 68, 48  # assumed 34x24 sprite, doubled
PIPE_W, PIPE_IMG_H = 104, 640  # assumed 52x320 sprite, doubled
GAP, PIPE_VEL = 200, 5
FLOOR_Y = 730
JUMP_VEL = -10.5
NEXT_PIPE_X = 420
SAFETY_FRAMES = 200000


class Bird:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vel = 0.0
        self.tick_count = 0

    def jump(self):
        self.vel = JUMP_VEL
        self.tick_count = 0

    def move(self):
        self.tick_count += 1
        d = self.vel * self.tick_count + 1.5 * self.tick_count ** 2
        if d >= 16:
            d = 16
        if d < 0:
            d -= 2
        self.y += d


class Pipe:
    def __init__(self, x, height):
        self.x = x
        self.height = height              # y of the top edge of the gap
        self.bottom = height + GAP        # y of the bottom edge of the gap
        self.top = height - PIPE_IMG_H    # y of the top-left of the top pipe sprite
        self.passed = False

    def collide(self, bird):
        if bird.x + BIRD_W <= self.x or bird.x >= self.x + PIPE_W:
            return False
        return bird.y < self.height or bird.y + BIRD_H > self.bottom


def step_frame(birds, pipes, choose, next_height, on_pass=None):
    """Advance one frame, mirroring the body of main() in flappy.py.

    `birds` is a list of (bird, tag) pairs. `choose(tag, bird, pipe)` returns True
    to jump. `on_pass(birds)` is called when a pipe is passed, after pipe
    collisions and BEFORE the floor check, as in flappy.py (a bird that dies on
    the floor in that same frame still gets the pass reward). Returns
    (tags_that_hit_a_pipe, tags_that_hit_the_floor, passed_pipe).
    """
    pipe_ind = 0
    if birds and len(pipes) > 1 and birds[0][0].x > pipes[0].x + PIPE_W:
        pipe_ind = 1
    for bird, tag in birds:
        bird.move()
    jumps = [choose(tag, bird, pipes[pipe_ind]) for bird, tag in birds]
    for (bird, tag), j in zip(birds, jumps):
        if j:
            bird.jump()
    hit, floor, add_pipe = [], [], False
    rem = []
    for pipe in pipes:
        pipe.x -= PIPE_VEL
        for bird, tag in list(birds):
            if pipe.collide(bird):
                hit.append(tag)
                birds.remove((bird, tag))
        if not pipe.passed and pipe.x < BIRD_X:
            pipe.passed = True
            add_pipe = True
        if pipe.x + PIPE_W < 0:
            rem.append(pipe)
    if add_pipe:
        if on_pass:
            on_pass(birds)
        pipes.append(Pipe(NEXT_PIPE_X, next_height()))
    for r in rem:
        pipes.remove(r)
    for bird, tag in list(birds):
        if bird.y + BIRD_H >= FLOOR_Y:
            floor.append(tag)
            birds.remove((bird, tag))
    return hit, floor, add_pipe


def play_policy(policy, heights, max_frames=2000):
    """Deterministic single-bird run used to cross-check other implementations.

    `policy(y, gap_top, gap_bottom)` -> bool (jump). `heights` is the list of gap
    heights, used in order (first for the opening pipe).
    Returns dict(frames, score, y).
    """
    it = iter(heights)
    pipes = [Pipe(WIN_WIDTH, next(it))]
    birds = [(Bird(BIRD_X, BIRD_Y0), 0)]
    score = frames = 0

    def choose(tag, bird, pipe):
        return policy(bird.y, pipe.height, pipe.bottom)

    def nxt():
        try:
            return next(it)
        except StopIteration:
            return 250

    while birds and frames < max_frames:
        frames += 1
        _, _, passed = step_frame(birds, pipes, choose, nxt)
        if passed:
            score += 1
    return {"frames": frames, "score": score}


def make_eval(state, threshold):
    def eval_genomes(genomes, config):
        state["gen"] += 1
        nets, ge, birds = {}, {}, []
        for gid, g in genomes:
            g.fitness = 0.0
            nets[gid] = neat.nn.FeedForwardNetwork.create(g, config)
            ge[gid] = g
            birds.append((Bird(BIRD_X, BIRD_Y0), gid))
        pipes = [Pipe(WIN_WIDTH, random.randrange(50, 450))]
        score = frames = 0

        def choose(gid, bird, pipe):
            ge[gid].fitness += 0.1  # staying alive, every frame
            out = nets[gid].activate((bird.y, abs(bird.y - pipe.height), abs(bird.y - pipe.bottom)))
            return out[0] > 0.5

        def nxt():
            return random.randrange(50, 450)

        def reward(alive):
            for _, gid in alive:
                ge[gid].fitness += 5

        while birds and frames < SAFETY_FRAMES:
            frames += 1
            hit, _, passed = step_frame(birds, pipes, choose, nxt, reward)
            for gid in hit:
                ge[gid].fitness -= 1
            if passed:
                score += 1
            if threshold is not None and any(ge[gid].fitness >= threshold for _, gid in birds):
                break
        state["scores"].append(score)
    return eval_genomes


class Recorder(neat.reporting.BaseReporter):
    """Collects best/mean fitness and species count after each evaluation."""

    def __init__(self):
        self.rows = []

    def post_evaluate(self, config, population, species, best_genome):
        fits = [g.fitness for g in population.values()]
        self.rows.append({"gen": len(self.rows), "best": round(max(fits), 2),
                          "mean": round(sum(fits) / len(fits), 2), "species": len(species.species)})


def train(config_path, seed, generations):
    random.seed(seed)
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    threshold = config.fitness_threshold
    pop = neat.Population(config)
    rec = Recorder()
    pop.add_reporter(rec)
    state = {"gen": 0, "scores": []}
    try:
        winner = pop.run(make_eval(state, threshold), generations)
    except neat.CompleteExtinctionException:
        winner = None
    for row, score in zip(rec.rows, state["scores"]):
        row["score"] = score
    reached = next((r["gen"] for r in rec.rows if r["best"] >= threshold), None)
    return {"seed": seed, "threshold": threshold, "reached_gen": reached,
            "generations_run": len(rec.rows), "per_gen": rec.rows,
            "winner_size": list(winner.size()) if winner else None}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=os.path.join(here, "config.txt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, default=20, help="flappy.py uses 20")
    ap.add_argument("--runs", type=int, default=1, help="repeat with seeds seed..seed+runs-1")
    ap.add_argument("--json", help="write all results here")
    args = ap.parse_args()

    results = [train(args.config, args.seed + i, args.generations) for i in range(args.runs)]
    if args.runs == 1:
        r = results[0]
        print("gen  best    mean   species  score(pipes)")
        for row in r["per_gen"]:
            print(f"{row['gen']:>3} {row['best']:>7.1f} {row['mean']:>7.1f} {row['species']:>5} {row['score']!s:>8}")
        print("threshold", r["threshold"], "reached at generation", r["reached_gen"])
    else:
        reached = [r["reached_gen"] for r in results if r["reached_gen"] is not None]
        print(f"{len(results)} runs, seeds {args.seed}..{args.seed + args.runs - 1}, up to {args.generations} generations")
        print(f"reached fitness {results[0]['threshold']:g}: {len(reached)}/{len(results)} runs")
        if reached:
            print("generation reached (0-based): median", statistics.median(reached),
                  "mean", round(statistics.mean(reached), 2), "min", min(reached), "max", max(reached))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
