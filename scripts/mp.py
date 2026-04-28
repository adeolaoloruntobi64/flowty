from coloraide import Color

color = Color("srgb", [1, 0, 0], 1)
okl = color.convert("oklab")
print(color, okl)

LST = [
    (69, 67, 35),
    (107, 104, 54),
    (98, 95, 50),
    (32, 32, 17),

    (255, 0, 0),
    (70, 9, 0),
    (56, 24, 0),
    (55, 27, 0),
    (58, 45, 0),
    (234, 224, 0),

    (72, 40, 8),
    (81, 50, 13),
    (79, 49, 12),
    (63, 33, 4)
]

LST2 = [
    (107, 104, 54),
    (28, 30, 28),
    (54, 55, 42)
]

for i in range(0, 3):
    (r, g, b) = LST2[i]
    pre = Color("srgb", (r / 255, g / 255, b / 255), 1)
    post = pre.convert("oklab")
    (l, c, h) = post.coords()
    l *= 100
    c *= 100
    if i == 4 or i == 10:
        print()
    print(f"{(r, g, b)} -> {(l, c, h)}")