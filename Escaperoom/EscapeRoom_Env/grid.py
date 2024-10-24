from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal, Key

class ColorGrid(Grid):
    def __init__(self, width, height):
        super().__init__(width, height)
        # Initialize a 2D list to store the color of each cell (including floors)
        self.grid_colors = [[None for _ in range(height)] for _ in range(width)]

    def set_color(self, x, y, color):
        """Set the color of a specific grid cell."""
        self.grid_colors[x][y] = color

    def get_color(self, x, y):
        """Get the color of a specific grid cell."""
        return self.grid_colors[x][y]