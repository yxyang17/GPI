from pusht.envs import PushTEnv, PushTImageEnv
import cv2
import numpy as np

env = PushTEnv()
obs, _ = env.reset()

render_video = False
live_display = True
capture_frames = render_video or live_display
initial_frame = env.render(mode="rgb_array") if capture_frames else None
frames: list[np.ndarray] = (
    [initial_frame] if (render_video and initial_frame is not None) else []
)
if live_display and initial_frame is not None:
    cv2.imshow(
        "PushT State Policy", cv2.cvtColor(initial_frame, cv2.COLOR_RGB2BGR)
    )
    cv2.waitKey(1)

for i in range(10):
    obs, reward, done, _, _ = env.step(obs[:2])
    print(obs)
    if capture_frames:
        frame = env.render(mode="rgb_array")
        if render_video:
            frames.append(frame)
        if live_display:
            cv2.imshow(
                "PushT State Policy", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
            cv2.waitKey()