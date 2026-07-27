import pygame
from src.config import COLOR_GRAY


class Track:

    def __init__(self):
        self.outer_points = [
            (100, 80),
            (900, 80),
            (950, 180),
            (950, 420),
            (900, 520),
            (100, 520),
            (50, 420),
            (50, 180),
        ]

        self.inner_points = [
            (200, 180),
            (800, 180),
            (850, 230),
            (850, 370),
            (800, 420),
            (200, 420),
            (150, 370),
            (150, 230),
        ]

        # List to store all wall line segments
        self.walls = []
        self._create_walls()

    def _create_walls(self):
        """Connects the coordinate points to form continuous line segments (walls)."""
        
        for i in range(len(self.outer_points)):
            pt1 = self.outer_points[i]
            # Connect the last point back to the first point to close the loop
            pt2 = self.outer_points[(i + 1) % len(self.outer_points)]
            self.walls.append((pt1, pt2))

        for i in range(len(self.inner_points)):
            pt1 = self.inner_points[i]
            pt2 = self.inner_points[(i + 1) % len(self.inner_points)]
            self.walls.append((pt1, pt2))

    def draw(self, display):
        """Renders the track walls onto the Pygame window."""
        for wall in self.walls:
            pygame.draw.line(display, COLOR_GRAY, wall[0], wall[1], 3)