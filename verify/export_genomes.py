"""Export evolved neat-python genomes plus their outputs on fixed inputs, as JSON.

verify/check_engine.js loads the same genomes into the JavaScript network and
checks that it gives the same outputs (feed-forward layering, tanh with the
2.5 gain, biases, disabled connections, hidden nodes).

    python verify/export_genomes.py > genomes.json
"""
import json
import os
import random
import sys

import neat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import headless_train as ht  # noqa: E402

INPUTS = [(310.0, 100.0, 300.0), (50.0, 12.5, 212.5), (600.0, 250.0, 50.0),
          (-20.0, 70.0, 270.0), (400.0, 0.0, 200.0)]


class Grab(neat.reporting.BaseReporter):
    def __init__(self):
        self.genomes = []

    def post_evaluate(self, config, population, species, best_genome):
        if len(self.genomes) < 400:
            self.genomes.extend(population.values())


def main():
    random.seed(5)
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                os.path.join(os.path.dirname(HERE), "config.txt"))
    pop = neat.Population(config)
    grab = Grab()
    pop.add_reporter(grab)
    state = {"gen": 0, "scores": []}
    pop.run(ht.make_eval(state, None), 40)
    out = []
    for g in grab.genomes[::7]:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        out.append({
            "nodes": [{"key": k, "bias": n.bias, "response": n.response, "activation": n.activation}
                      for k, n in g.nodes.items()],
            "conns": [{"i": k[0], "o": k[1], "weight": c.weight, "enabled": c.enabled}
                      for k, c in g.connections.items()],
            "inputs": [list(i) for i in INPUTS],
            "outputs": [net.activate(i)[0] for i in INPUTS],
        })
    print(json.dumps(out))


if __name__ == "__main__":
    main()
