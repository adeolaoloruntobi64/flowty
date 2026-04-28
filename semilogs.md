Thss should be gone after the first commit
I am leaving it in the dirst commit to preserve the history of my thoughts



An app that should be able to solve puzzles from:

Flow Free
Flow Free Bridges
Flow Free Hexes
Flow Free Warps
Flow Free Shapes

Resources:
https://torvaney.github.io/projects/flow-solver.html
https://mzucker.github.io/2016/08/28/flow-solver.html
https://mzucker.github.io/2016/09/02/eating-sat-flavored-crow.html
https://github.com/mzucker/flow_solver/blob/master/pyflowsolver.py
https://github.com/laggycomputer/permanganate/tree/master
https://arxiv.org/html/2505.15221v1#S3.T1
https://github.com/leophagus/Flow-Free-Solver/tree/master
https://www.youtube.com/watch?v=XU4Xk_zg9jI
https://www.samueltgoldman.com/post/flow-solver/

Next: Find a way to generalize the solver.
Arbitrary graph/shape (hex, triangle, pentagon + triangle, figure 8)
Arbitrary warps (maybe a war goes from top left to botton right)
Arbitrary bridges (same as just chanining bridges)
Arbitrary edges (cutting off a connection)

Next: Find a way to detect and solve a flow game

Goated
https://stackoverflow.com/questions/37517983/opencv-install-opencv-contrib-on-windows

TOPHAT is the GOAT
(3, 3) rect + open 2 iter, morph got all the circles
(3, 3) rect + tophat 4 iter, morph gets the entire grid
(3, 15) rect + tophat 4 iter, morph gets the entire grid and all the circles

Here holds the number of times I quit, and the duration of the quit
Note, I started documenting this at quit #5, which as of writing, has not happened
yet. I came back only a couple days ago from quit #4, which was almost permanent
if not for a late night thought

1. Canny edge detection doesn't capture everything properly (5 days)
2. Sobel is too blurry, but Canny is too inaccurate (4 days)
3. The weighted sum of Canny and Sobel has instances is really stupid. The alpha
was originally at 0.6, but it broke for some things. After a lot of trial and
error, 0.59 worked, but that broke another thing. After that, 0.596 seemed to work
until it didn't. It's also exam season (29 days)
4. I switched to morphology, and discovered Tophat by accident. Seemed like a
gift from God. It was perfect. And then everything went wrong. I just found out
about alley levels, and on one of the levels, the line is too dim and dissapears
because of my thresholding, but I can't afford to lower the threshold, so instead,
I'm dilating then eroding. But this causes a new issue. Morph Tophat basically
gets rid of the dots, but not the rings that ONLY appear in chain levels, and when
I would dilate and erode, the rings would touch the boundary of the cell and break
the cell into multiple parts. I spent weeks trying to do a combination of morphology
transformations to get rid of the rings, even tried using ximgproc fit_ellipses,
tried to use distance_transform to create a dilation algorithm that took the size
of the object into consideration somehow, but nothing worked. This was the quit
I thought would be permanent (22 days)
    - This was supposed to be the end of my endeavors, but one night, I realized that
    the ring was always a dimmer version of the dot. And when looking at it in paint,
    I noticed the RGB seemed to be multiplied by a constant factor, ~0.69 from experiments.
    Then I saw Paint had an option to view the pixel in HSV and noticed that while H and S
    stayed the same (S = 100), V was multiplied by ~0.695 (V = 100 -> 69). And thus, the
    new pipeline was born. I hypothesize #5 will be about warps, but I'll be glad if it isn't
5. Finals (21 days)
6. This is after finals. I realized that supporting all chain and alley levels is basically
    impossible with just OpenCV operations. I'm not sure how to feel. I remember starting this
    project in November 2025. It is now April 2026. To whoever reads this, I spent ~5 months
    trying to perfect the image recognition, while the screen capturing + board solving + mouse
    input COMBINED took me less that 24 hours. Quite a bit less actually. Every time I think
    I'll find a solution the next day, but I never do. This project is the reason I do not have
    a 4.0 this semester. Not to say my GPA is bad, it definently isn't. But it is no longer the
    supremum. I plan to put away this project before the next semester starts, whether it's done or
    not. Maybe I will find a way to do chains or warps. If I could detect them, the rest of the
    program can handle them no problem. But for now, Chains and Warps are unsupported. I was going
    to train a Instance Segmentation model to recognize cells, but I now have 417 images throughout
    all 5 flow free games that I'd have to label, and I'm not sure if it'd be worth it. For all I
    could spend another 6 months all for nothing. All my attempts will be in a "saves" folder on
    first commit, then I will delete a bunch of stuff on the second commit.


Before I forget
- I forked rustautogui because I wanted to take a screenshot without saving to disk.
- I tried to use scap but it didn't build. But then I found xcap
- I didn't use most of the features of rustautogui anyways, so I swapped it out for xcap + enigo
- I forked cmake because when I started, it didn't support vs2026. It didn't for a while so I gave up waiting
- The Permanganate Crate (https://docs.rs/permanganate/latest/permanganate/) is the inspiration for my solver.
    However the Shape and FullShape traits were too specific/restrictive for my use case, so I just took
    the solver from that and changed the SAT solver to use the rustsat crate since it can generalize over solvers. Additionally, I made a change to the exactly_one function that improved the runtime by 4x.

I just had a realization
The reason why chains weren't supported was because of dilation
I dilated to fix the warp lines
Since I decided not to support chains and warps, I removed the dilation
Allowing me to solve chains
I wouldn't have realized this if I didn't test with a chain level by accident
and notice it worked
Oh my days
All my problems exist cuz of warps

Keep
87.8   -10.1   85.7
54.3    80.8   67.9
50.8   -51.0   52.1
69.0    35.7   75.1
33.3    57.5 -105.2
47.9   -70.3   49.0
43.8    -4.5   28.3
27.2     1.7   18.8

43.8    -4.5   28.3
18.2     2.9   10.8
17.3     4.1   19.0
22.1    42.7  -79.4

29.7    56.1  -36.3
71.3    44.4   20.5
35.0    27.8   13.1
27.5    22.5   10.5
66.3     4.7  -15.5
90.7   -50.7  -15.8

41.8   -24.3   20.6
38.4   -23.0   19.5
86.7   -45.0   38.8
14.7   -10.7    8.9
18.9   -13.3   11.1
15.5   -11.2    9.3
-16.8  -12.3   10.3

44.8    -3.7   25.7

Discard
10.6   -15.6  -46.6
6.7     25.9  -11.1
8.3     16.5  -14.0
12.8    18.2  -17.7
16.4    10.7  -19.7
16.8    -2.3  -14.5

12.0     8.1   10.6
15.3     7.7   15.0
9.7      8.9    7.1
15.1     0.8   18.7
14.4    -3.5   20.1

3.3     16.0  -11.0
2.3      9.5   -5.1
14.4   -14.4   -4.3
8.4    -16.0   12.1
9.6     29.4  -12.3
16.2    30.2  -12.9
15.7   -11.4   -7.9

6.2     14.9    5.5

cell dot bridge ring warp board chain

Detecting cells
------

When analyzing the rings in chain puzzles, I noticed they were
always darker. Then I put an image in paint and saw that for a
blue dot (12 41 254), the ring was (8 28 177). And for the green
dot (0 141 0), the green ring was (0 98 0). 98/141 is approximately
0.695, and sure enough, 0.695 * (12 41 254) = (8.34 28.495 176.53)
= (8, 28, 177) when rounded to the nearest whole number. This is further
confirmed by looking at their hsv values. Actually, a ring will always
have the same h and s value as a dot, but the v is multiplied by 0.69
or 0.695 depending on the case. But it's definently not 0.7.
const CENTER_TO_RING: f64 = 0.695; // We don't need this anymore.

When implementing the image recognition, I ran into a problem for
an alley level. Because of our thresholding, some of the grid lines
go missing, and lowering the threshold messes up the logic that allows
this to work for the mountain levels. So instead, we preprocess by
dilating then eroding to connect missing lines. To be extra careful,
we also need to get rid of the small artifacts of the dots that tophat
leaves behind

Sort by descending area so that cell comes before dot
Later in development, we don't really care about identifying the dot
But keep this just in case we do detect it
Even later, the dots are removed, so remove this if necessary
// clusters.sort_by(|a, b| b.area.total_cmp(&a.area));

Expanding the bounding box for ROI:
We take the bounding box of a cell (rect) and expand it by DILATION_PIXELS
in all directions. When we dilate the cell's edges later, we don't get
clipping at the edges of the ROI. Then we intersect with `imgbound`
(the full image bounds) to ensure the expanded rect stays inside the image.
This leaves us with a slightly larger ROI that fully contains the cell
and any (to be) dilated edges.

Edge dilation adjacency check:
Draw each cell's edges as a thin binary mask. Then dilate edges by
'DILATION_PIXELS' pixels using an elliptical kernel. This creates a small
buffer around edges to bridge gaps. We specifically use an elliptical kernel
to reduces the amount diagonal contact overlap. Then we draw candidate neighbor
cell's edges as a thin line. Then we bitwise AND between dilated edges and
candidate edges. The idea is we check how many pixels of the edges of cell
b are in the dilation area of cell a. This also makes it so that corners don't
qualify as a neighbor as there wouldn't be enough of those edge pixels in the
dilation area. If overlap >= minSharedPixels, then the 2 cells are neighbors.
Since we're no longer using them, why not reuse them

Keep a list of almost neighbors (this was when I didn't have them yet)
If len(almost) / len(neighbors + almost) > 0.6 (0.66 is 2/3 for triangle)
This means it is mostly surrounded by boundary edges
if ~ 2/3 check for mountain (0.61-0.70)
if ~3/4 check for underpass (0.70-0.79)
(60, 80) exclusive
If we've already calculated, then continue
Now that I'm looking at it, the triangles on mountain have 1 neighbor, and 1
almost neighbor. So 50%. underpass is still 3/4

Made a table and discovered
1. Saturation of the circles are high
2. Blobs have high saturation and low values

| S   | V   | Result |
| --- | --- | ------ |
| 100 | 100 | y      |
| 100 | 50  | y      |
| 75  | 64  | y      |
| 0   | 100 | y      |
| 16  | 74  | y      |
| 100 | 28  | n      |
| 100 | 33  | n      |
| 93  | 23  | n      |
| 96  | 20  | n      |
| 100 | 15  | n      |
| 49  | 40  | y      |
| 45  | 88  | y      |
| 75  | 25  | ~      |
| 51  | 37  | y      |
| 43  | 18  | ~      |
| 41  | 92  | y      |
Seems I can apply the rule IF (S >= 90) AND (V <= 35) then it's a blob
or, for hsv_full
V <= 40 OR (S >= 230 AND V <= 90)
V <= 40 OR (S >= 210 AND V <= 90)

If this was numberlink, tophat would be good enough.
The only reason why I can't use tophat is because of the rings in chain levels

I need an algorithm that does this

    bw: &mut Mat,
    max_back: i32,
    min_run: i32,
    max_forward: i32,
    forward_presence_radius: i32,
    directions: &[(i32, i32)]

UI Manager
Image Processing (Mostly done)
Solver (Mostly done)
Drawer