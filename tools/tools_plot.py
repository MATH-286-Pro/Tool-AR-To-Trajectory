import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _set_equal_3d_axes(ax, xyz):
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    center = mins + (maxs - mins) / 2
    radius = np.max(maxs - mins) / 2
    if radius < 1e-8:
        radius = 1e-3

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")


def _apply_3d_view(ax, view):
    match view.lower():
        case "normal":
            pass
        case "top":
            ax.view_init(elev=90, azim=-180)
        case "side":
            ax.view_init(elev=0, azim=-90)
        case "front":
            ax.view_init(elev=0, azim=0)
        case _:
            raise ValueError("3D view must be 'normal', 'top', 'side', or 'front'")


def _axis_angle_to_rot(axis_angle):
    axis_angle = np.asarray(axis_angle, dtype=float)
    if axis_angle.shape != (3,):
        raise ValueError(f"axis_angle must have shape (3,), got {axis_angle.shape}")

    angle = np.linalg.norm(axis_angle)
    if angle < 1e-12:
        return np.eye(3)

    x, y, z = axis_angle / angle
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1 - c

    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def _select_trajectory(data, index):
    if isinstance(data, (str, Path)):
        with open(data, "rb") as f:
            data = pickle.load(f)

    if isinstance(data, dict):
        return data

    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        return arr

    if index is None:
        index = 0
    if index < 0 or index >= len(data):
        raise IndexError(f"index must be in [0, {len(data) - 1}]")
    return data[index]


def _extract_trajectory_pose(trajectory):
    if isinstance(trajectory, dict):
        if "ee_tf" in trajectory:
            tf = np.asarray(trajectory["ee_tf"], dtype=float)
            if tf.ndim != 3 or tf.shape[1:] != (4, 4):
                raise ValueError("trajectory['ee_tf'] must have shape (T, 4, 4)")
            pos = tf[:, :3, 3]
            rot = tf[:, :3, :3]
        else:
            pos = np.asarray(trajectory["ee_pos"], dtype=float)
            if pos.ndim != 2 or pos.shape[1] != 3:
                raise ValueError("trajectory['ee_pos'] must have shape (T, 3)")
            axis_angle = trajectory.get("ee_axis_angle")
            rot = None
            if axis_angle is not None:
                axis_angle = np.asarray(axis_angle, dtype=float)
                if axis_angle.shape != pos.shape:
                    raise ValueError("trajectory['ee_axis_angle'] must have shape (T, 3)")
                rot = np.stack([_axis_angle_to_rot(a) for a in axis_angle], axis=0)

        t = np.asarray(trajectory.get("t", np.arange(len(pos))), dtype=float)
    else:
        tf = np.asarray(trajectory, dtype=float)
        if tf.ndim != 3 or tf.shape[1:] != (4, 4):
            raise ValueError("trajectory array must have shape (T, 4, 4)")
        pos = tf[:, :3, 3]
        rot = tf[:, :3, :3]
        t = np.arange(len(pos), dtype=float)

    if len(pos) == 0:
        raise ValueError("trajectory is empty")
    if len(t) != len(pos):
        t = np.arange(len(pos), dtype=float)

    return pos, rot, t


def plot_3d(
    tf,
    arrow_length=0.1,
    interval: float = 0.1,
    view: str = "normal",
):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tf[:, 0, 3], tf[:, 1, 3], tf[:, 2, 3], "k-", label="Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 绘制 orientation
    for i in range(0, len(tf), int(len(tf) * interval)):
        T = tf[i]
        origin = T[:3, 3]
        x_axis = origin + T[:3, 0] * arrow_length
        y_axis = origin + T[:3, 1] * arrow_length
        z_axis = origin + T[:3, 2] * arrow_length
        ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color="r")
        ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color="g")
        ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color="b")

    _apply_3d_view(ax, view)

    xyz = tf[:, :3, 3]
    _set_equal_3d_axes(ax, xyz)

    plt.show()


def plot_traj_gif(
    trajectory,
    *,
    index: int | None = None,
    save_path="trajectory.gif",
    view: str = "top",
    fps: int = 20,
    stride: int = 2,
    axis_len: float = 0.04,
    show_orientation: bool = True,
    show_frame_history: bool = False,
    frame_history_stride: int = 10,
    frame_history_alpha: float = 0.25,
    dpi: int = 120,
    title: str | None = None,
    display_gif: bool = True,
):
    """Save an end-effector trajectory as a GIF animation.

    ``trajectory`` can be a single trajectory dict, a list of trajectory dicts
    with ``index``, a pickle path containing a trajectory list, or a raw
    transform array with shape ``(T, 4, 4)``. Dict inputs can contain either
    ``ee_tf`` or ``ee_pos`` plus optional ``ee_axis_angle``. Coordinate-frame
    axes are drawn as X=red, Y=green, Z=blue.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    trajectory = _select_trajectory(trajectory, index)
    traj, rot, t = _extract_trajectory_pose(trajectory)

    fps = int(fps)
    if fps <= 0:
        raise ValueError("fps must be positive")

    stride = max(1, int(stride))
    frame_history_stride = max(1, int(frame_history_stride))
    frame_ids = list(range(0, len(traj), stride))
    if frame_ids[-1] != len(traj) - 1:
        frame_ids.append(len(traj) - 1)

    show_orientation = show_orientation and rot is not None
    show_frame_history = show_frame_history and show_orientation

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="0.85", linewidth=1.0)
    path_line, = ax.plot([], [], [], color="tab:blue", linewidth=2.0)
    current_point = ax.scatter([], [], [], c="tab:orange", s=45)
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c="g", marker="o", s=35, label="start")
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c="r", marker="x", s=35, label="end")
    time_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)
    frame_label = "frame axes: X=red, Y=green, Z=blue" if show_orientation else ""
    frame_text = ax.text2D(0.03, 0.90, frame_label, transform=ax.transAxes)

    _set_equal_3d_axes(ax, traj)
    _apply_3d_view(ax, view)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title or "Trajectory")
    ax.legend(loc="upper right")

    frame_artists = []

    def draw_frame(frame_id, *, length, alpha=1.0, linewidth=1.8, label_axes=False):
        p = traj[frame_id]
        rotation = rot[frame_id]
        artists = []
        for axis_id, color in enumerate(("r", "g", "b")):
            direction = rotation[:, axis_id]
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction = direction / norm
            artists.append(
                ax.quiver(
                    p[0], p[1], p[2],
                    direction[0], direction[1], direction[2],
                    length=length,
                    normalize=True,
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                )
            )
            if label_axes:
                endpoint = p + direction * length * 1.15
                artists.append(
                    ax.text(
                        endpoint[0], endpoint[1], endpoint[2],
                        ("X", "Y", "Z")[axis_id],
                        color=color,
                        fontsize=9,
                    )
                )
        return artists

    def update(frame_id):
        nonlocal frame_artists
        p = traj[frame_id]
        passed = traj[: frame_id + 1]

        path_line.set_data(passed[:, 0], passed[:, 1])
        path_line.set_3d_properties(passed[:, 2])
        current_point._offsets3d = ([p[0]], [p[1]], [p[2]])
        time_text.set_text(f"frame {frame_id + 1}/{len(traj)}  t={t[frame_id]:.2f}s")

        for artist in frame_artists:
            artist.remove()
        frame_artists = []

        if show_orientation:
            if show_frame_history:
                for hist_id in range(0, frame_id + 1, frame_history_stride):
                    frame_artists.extend(
                        draw_frame(
                            hist_id,
                            length=axis_len * 0.65,
                            alpha=frame_history_alpha,
                            linewidth=1.0,
                            label_axes=False,
                        )
                    )
            frame_artists.extend(
                draw_frame(
                    frame_id,
                    length=axis_len,
                    alpha=1.0,
                    linewidth=2.0,
                    label_axes=True,
                )
            )

        return [path_line, current_point, time_text, frame_text, *frame_artists]

    ani = FuncAnimation(fig, update, frames=frame_ids, interval=1000 / fps, blit=False)
    ani.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)

    if display_gif:
        try:
            from IPython.display import Image, display
            display(Image(filename=str(save_path)))
        except Exception:
            pass

    return save_path
