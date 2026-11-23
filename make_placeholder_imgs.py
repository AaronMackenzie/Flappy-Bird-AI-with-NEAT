"""Write simple placeholder sprites into imgs/ so flappy.py can run.

The original art is not in this repository. flappy.py expects these files,
which it doubles with pygame.transform.scale2x:

    bird1.png bird2.png bird3.png   34 x 24
    pipe.png                        52 x 320
    bg.png                          288 x 512
    base.png                        336 x 112

If you have the real sprites, copy them into imgs/ instead and skip this
script. Existing files are never overwritten.

    python make_placeholder_imgs.py
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs")


def bird(wing_y):
    s = pygame.Surface((34, 24), pygame.SRCALPHA)
    pygame.draw.ellipse(s, (242, 184, 75), (0, 0, 34, 24))
    pygame.draw.ellipse(s, (200, 140, 40), (6, wing_y, 14, 9))
    pygame.draw.circle(s, (255, 255, 255), (25, 8), 5)
    pygame.draw.circle(s, (20, 20, 20), (26, 8), 2)
    pygame.draw.polygon(s, (224, 97, 58), [(31, 12), (34, 14), (31, 17)])
    return s


def pipe():
    s = pygame.Surface((52, 320))
    s.fill((43, 127, 117))
    pygame.draw.rect(s, (95, 209, 180), (4, 0, 6, 320))
    pygame.draw.rect(s, (31, 95, 87), (0, 0, 52, 13))
    return s


def bg():
    s = pygame.Surface((288, 512))
    for y in range(512):
        k = y / 511
        pygame.draw.line(s, (int(112 + 30 * k), int(197 - 20 * k), int(206 - 10 * k)), (0, y), (287, y))
    return s


def base():
    s = pygame.Surface((336, 112))
    s.fill((222, 216, 149))
    pygame.draw.rect(s, (115, 191, 46), (0, 0, 336, 12))
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    pygame.init()
    files = {"bird1.png": bird(10), "bird2.png": bird(7), "bird3.png": bird(13),
             "pipe.png": pipe(), "bg.png": bg(), "base.png": base()}
    for name, surf in files.items():
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            print("kept existing", path)
        else:
            pygame.image.save(surf, path)
            print("wrote", path)


if __name__ == "__main__":
    main()
