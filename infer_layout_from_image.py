from rl_config import *
from RoomLayoutEstimator import RoomLayoutEstimator


def main():
    # Perform room layout estimation
    room_layout_estimator = RoomLayoutEstimator(example=ex)
    room_layout_estimator.infer_layout()

    # Visualize results
    room_layout_estimator.visualize_layout_2d()
    room_layout_estimator.visualize_layout_3d()



if __name__ == "__main__":
    main()
