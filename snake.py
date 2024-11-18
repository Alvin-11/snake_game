from turtle import Turtle
starting_positions = [(0, 0), (-20, 0), (-40, 0)]
move_distance = 20
up = 90
down = 270
left = 180
right = 0
class Snake:

    def __init__(self): # initialize the variables
        self.segments = [] # keeps hold of the position of each cube representing the snake
        self.create_snake()
        self.head = self.segments[0]


    def create_snake(self): # creates the snake whenever the game start or is reset
      for position in starting_positions:
        self.add_segment(position)

    def add_segment(self,position): # add another length to the snake when it eats the food.
        new_segment = Turtle("square")
        new_segment.color("white")
        new_segment.penup()
        new_segment.goto(position)
        self.segments.append(new_segment)

    def reset(self): # reset the entierety of the game once the player lose
        for seg in self.segments:
            seg.goto(1000,1000)
        self.segments.clear()
        self.create_snake()
        self.head = self.segments[0]

    def extend(self): # Extends the length of the snake
        self.add_segment(self.segments[-1].position())

    def move(self): # This makes the snake move forward
        for seg_num in range(len(self.segments) - 1, 0, -1):
            new_x = self.segments[seg_num - 1].xcor()
            new_y = self.segments[seg_num - 1].ycor()
            self.segments[seg_num].goto(new_x, new_y)

        self.segments[0].forward(move_distance)

    def up(self): # This makes the snake move up
        if self.head.heading() != down:
         self.head.setheading(up)

    def down(self): # This makes the snake move down
        if self.head.heading() != up:
         self.head.setheading(down)

    def left(self): # This makes the snake move left
        if self.head.heading() != right:
            self.head.setheading(left)

    def right(self): # This makes the snake move right
        if self.head.heading() != left:
            self.head.setheading(right)