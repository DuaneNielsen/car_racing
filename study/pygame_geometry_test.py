from pygame_geometry.abstract import Point, Circle, Segment
from pygame_geometry.context import Context
from pygame_geometry import colors

context = Context(name="title") # create a context similar to a pygame surface

p1 = Point(2,2)
p2 = Point(3,2, color=colors.BLUE)
c = Circle(0, -1, radius=2, color=colors.RED)
seg = Segment(Point(3, 3), Point(-3, 3))

# main game loop
while context.open:

    # clear the window
    context.clear()
    # check quit event (empty pygame event buffer by doing so)
    context.check()
    # move and zoom around the scene
    context.control()

    # update objects
    p1.rotate(0.01, p2)
    c.x += 0.01
    seg.rotate(0.01, p2)

    # show objects
    p1.show(context)
    p2.show(context)
    seg.show(context)
    c.show(context)

    # flip the screen
    context.flip()