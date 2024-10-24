from __future__ import annotations

from EscapeRoom_Env.env import EscapeRoomEnv
from minigrid.manual_control import ManualControl

def main():
    env = EscapeRoomEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=8)
    manual_control.start()

    
if __name__ == "__main__":
    main()
