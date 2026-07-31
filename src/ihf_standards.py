"""
IHF Official Handball Court Dimensions (Meters)
Reference: IHF Rules of the Game, Rule 1: The Court
"""

class IHFHandballCourt:
    # Overall Dimensions
    WIDTH = 40.0   # Sideline
    HEIGHT = 20.0  # Goal line
    
    # Goal Dimensions
    GOAL_WIDTH = 3.0
    GOAL_HEIGHT = 2.0
    
    # Area Lines
    GOAL_AREA_RADIUS = 6.0        # 6m line
    FREE_THROW_RADIUS = 9.0       # 9m line
    PENALTY_MARK_DISTANCE = 7.0   # 7m spot
    GK_LIMIT_LINE = 4.0           # 4m line
    
    # Zones (Normalized 0.0 to 1.0 for mapping)
    # Assuming center of the court is (20.0, 10.0)
    SUBSTITUTION_AREA_WIDTH = 4.5 # From center line
    
    @staticmethod
    def get_standard_corners():
        # Returns the 4 corners of the court in meters
        return [(0, 0), (40, 0), (40, 20), (0, 20)]