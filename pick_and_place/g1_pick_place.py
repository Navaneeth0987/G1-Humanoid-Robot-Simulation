"""
G1 Humanoid Pick & Place — MuJoCo (Fixed Base + Weld Grasp)
Run: python3 g1_pick_place.py
"""

import os
import mujoco
import mujoco.viewer
import numpy as np

ROBOT_DIR  = "/home/navaneeth/unitree_mujoco/unitree_robots/g1"
VIEWER_FPS = 60

# Joint indices (0-28)
L_HP=0; L_HR=1; L_HY=2; L_K=3;  L_AP=4;  L_AR=5
R_HP=6; R_HR=7; R_HY=8; R_K=9;  R_AP=10; R_AR=11
W_Y=12; W_R=13; W_P=14
L_SP=15; L_SR=16; L_SY=17; L_EL=18; L_WR=19; L_WP=20; L_WY=21
R_SP=22; R_SR=23; R_SY=24; R_EL=25; R_WR=26; R_WP=27; R_WY=28

def load_model():
    scene_path = os.path.join(ROBOT_DIR, "scene_29dof.xml")
    robot_path = os.path.join(ROBOT_DIR, "g1_29dof.xml")

    with open(scene_path, "r") as f:
        scene_xml = f.read()
    with open(robot_path, "r") as f:
        robot_xml = f.read()

    # Remove freejoint for fixed base
    robot_patched = robot_xml.replace(
        '<joint name="floating_base_joint" type="free" limited="false" actuatorfrclimited="false" />',
        ''
    )
    robot_tmp = os.path.join(ROBOT_DIR, "_g1_fixed.xml")
    with open(robot_tmp, "w") as f:
        f.write(robot_patched)

    scene_patched = scene_xml.replace(
        '<include file="g1_29dof.xml"/>',
        '<include file="_g1_fixed.xml"/>'
    )

    objects = """
    <!-- TABLE — taller legs (0.35 half-height = 0.70m tall) -->
    <body name="leg_fl" pos="0.65  0.13 0.35">
      <geom type="cylinder" size="0.025 0.35" rgba="0.55 0.37 0.20 1" contype="1" conaffinity="1"/>
    </body>
    <body name="leg_fr" pos="0.65 -0.13 0.35">
      <geom type="cylinder" size="0.025 0.35" rgba="0.55 0.37 0.20 1" contype="1" conaffinity="1"/>
    </body>
    <body name="leg_bl" pos="0.45  0.13 0.35">
      <geom type="cylinder" size="0.025 0.35" rgba="0.55 0.37 0.20 1" contype="1" conaffinity="1"/>
    </body>
    <body name="leg_br" pos="0.45 -0.13 0.35">
      <geom type="cylinder" size="0.025 0.35" rgba="0.55 0.37 0.20 1" contype="1" conaffinity="1"/>
    </body>
    <body name="table_top" pos="0.55 0.0 0.715">
      <geom type="box" size="0.13 0.16 0.018" rgba="0.76 0.60 0.42 1" contype="1" conaffinity="1"/>
    </body>
    <!-- RED CUBE — original size -->
    <body name="red_box" pos="0.55 0.0 0.775">
      <freejoint name="red_box_free"/>
      <geom type="box" size="0.050 0.050 0.050"
            rgba="0.95 0.10 0.10 1" mass="0.10" contype="1" conaffinity="1"/>
      <geom type="sphere" size="0.012" pos="0 0 0.053"
            rgba="1.0 0.95 0.0 1" contype="0" conaffinity="0"/>
    </body>
    <!-- GREEN TARGET ZONE on floor -->
    <body name="target_zone" pos="0.30 -0.70 0.005">
      <geom type="box" size="0.07 0.07 0.005"
            rgba="0.10 0.90 0.20 0.70" contype="0" conaffinity="0"/>
    </body>
"""
    weld = """
  <equality>
    <weld name="grasp_weld_right"
          body1="right_wrist_yaw_link"
          body2="red_box"
          active="false"
          relpose="0.10 0.06 0 1 0 0 0"/>
    <weld name="grasp_weld_left"
          body1="left_wrist_yaw_link"
          body2="red_box"
          active="false"
          relpose="0.10 -0.06 0 1 0 0 0"/>
  </equality>
"""
    scene_patched = scene_patched.replace("</worldbody>", objects + "\n  </worldbody>", 1)
    scene_patched = scene_patched.replace("</mujoco>", weld + "\n</mujoco>", 1)

    scene_tmp = os.path.join(ROBOT_DIR, "_scene_fixed.xml")
    with open(scene_tmp, "w") as f:
        f.write(scene_patched)

    try:
        model = mujoco.MjModel.from_xml_path(scene_tmp)
    finally:
        for p in [robot_tmp, scene_tmp]:
            if os.path.exists(p):
                os.remove(p)
    return model

# ── PD gains ──
KP = np.array([
    300,300,300, 300,300,300,
    300,300,300, 300,300,300,
    300,300,300,
    100,100,100, 100, 40, 40, 40,
    100,100,100, 100, 40, 40, 40,
], dtype=float)

KD = np.array([
     15, 15, 15,  15, 15, 15,
     15, 15, 15,  15, 15, 15,
     15, 15, 15,
      6,  6,  6,   6,  3,  3,  3,
      6,  6,  6,   6,  3,  3,  3,
], dtype=float)

TORQUE_LIMIT = np.array([
     88, 88, 88, 139, 50, 50,
     88, 88, 88, 139, 50, 50,
     88, 50, 50,
     25, 25, 25,  25, 25,  5,  5,
     25, 25, 25,  25, 25,  5,  5,
], dtype=float)

def pd_torque(data, q_des):
    q  = data.qpos[0:29]
    dq = data.qvel[0:29]
    return np.clip(KP*(q_des-q) - KD*dq, -TORQUE_LIMIT, TORQUE_LIMIT)

# ── Weld helpers ──
def enable_weld(model, data):
    """Weld red_box to BOTH wrists — simulates two-hand grasp."""
    for name in ["grasp_weld_right", "grasp_weld_left"]:
        wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if wid >= 0:
            model.eq_active0[wid] = 1
            data.eq_active[wid]   = 1
    print("    [GRASP] Box welded to both wrists ✓")

def disable_weld(model, data):
    """Release box from both wrists."""
    for name in ["grasp_weld_right", "grasp_weld_left"]:
        wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if wid >= 0:
            model.eq_active0[wid] = 0
            data.eq_active[wid]   = 0
    print("    [RELEASE] Box released ✓")

# ── Poses ──
def q_stand():
    q = np.zeros(29)
    q[L_HP]=-0.1; q[L_K]=0.2;  q[L_AP]=-0.1
    q[R_HP]=-0.1; q[R_K]=0.2;  q[R_AP]=-0.1
    q[L_SR]= 0.1; q[L_EL]=0.2
    q[R_SR]=-0.1; q[R_EL]=0.2
    return q

def q_reach():
    """Both arms reach forward and slightly outward to box sides — with forward lean."""
    q = q_stand().copy()
    q[W_P]= 0.25
    q[L_SP]=-1.20; q[L_SR]= 0.35; q[L_EL]=0.60; q[L_WP]=-0.10; q[L_WY]= 0.20
    q[R_SP]=-1.20; q[R_SR]=-0.35; q[R_EL]=0.60; q[R_WP]=-0.10; q[R_WY]=-0.20
    return q

def q_grasp():
    """Both arms close inward gripping sides of box — lower, leaning in."""
    q = q_reach().copy()
    q[W_P]= 0.30
    q[L_SP]=-1.35; q[L_SR]= 0.20; q[L_EL]=1.20; q[L_WP]=0.45; q[L_WY]= 0.20
    q[R_SP]=-1.35; q[R_SR]=-0.20; q[R_EL]=1.20; q[R_WP]=0.45; q[R_WY]=-0.20
    return q

def q_close():
    """Arms squeeze inward around box just before weld."""
    q = q_grasp().copy()
    q[L_SR]= 0.08; q[L_WY]= 0.10
    q[R_SR]=-0.08; q[R_WY]=-0.10
    return q

def q_lift():
    """Lift box up — straighten back up as arms rise."""
    q = q_stand().copy()
    q[W_P]= 0.10
    q[L_SP]=-1.00; q[L_SR]= 0.20; q[L_EL]=0.50
    q[R_SP]=-1.00; q[R_SR]=-0.20; q[R_EL]=0.50
    return q

def q_turn():
    """Waist turns right carrying box."""
    q = q_lift().copy()
    q[W_Y]=-1.20
    return q

def q_place():
    """Lower arms to place box on floor target."""
    q = q_turn().copy()
    q[L_SP]=-0.90; q[L_EL]=0.90; q[L_WP]=0.20
    q[R_SP]=-0.90; q[R_EL]=0.90; q[R_WP]=0.20
    return q

# ── Motion ──
def smooth_move(model, data, viewer, q_target, duration_s):
    q_start    = data.qpos[0:29].copy()
    n_steps    = int(duration_s / model.opt.timestep)
    sync_every = max(1, int((1.0/VIEWER_FPS)/model.opt.timestep))
    for i in range(n_steps):
        alpha        = 0.5*(1.0-np.cos(np.pi*i/n_steps))
        data.ctrl[:] = pd_torque(data, q_start + alpha*(q_target-q_start))
        mujoco.mj_step(model, data)
        if i % sync_every == 0 and viewer.is_running():
            viewer.sync()

def hold(model, data, viewer, q_des, duration_s):
    n_steps    = int(duration_s / model.opt.timestep)
    sync_every = max(1, int((1.0/VIEWER_FPS)/model.opt.timestep))
    for i in range(n_steps):
        data.ctrl[:] = pd_torque(data, q_des)
        mujoco.mj_step(model, data)
        if i % sync_every == 0 and viewer.is_running():
            viewer.sync()

# ── Sequence ──
def run_sequence(model, data, viewer):
    # Reset box to table at start of each loop
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_box")
    jnt_id = model.body_jntadr[box_id]
    qadr   = model.jnt_qposadr[jnt_id]
    data.qpos[qadr:qadr+3] = [0.55, 0.0, 0.775]  # back on table
    data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]       # upright
    data.qvel[model.jnt_dofadr[jnt_id]:model.jnt_dofadr[jnt_id]+6] = 0
    disable_weld(model, data)
    mujoco.mj_forward(model, data)
    print("  [1/7] Standing...")
    smooth_move(model, data, viewer, q_stand(), 2.0)
    hold(model, data, viewer, q_stand(), 1.0)

    print("  [2/7] Reaching forward to box...")
    smooth_move(model, data, viewer, q_reach(), 2.5)
    hold(model, data, viewer, q_reach(), 0.5)

    print("  [3/8] Grasping box...")
    smooth_move(model, data, viewer, q_grasp(), 1.5)
    hold(model, data, viewer, q_grasp(), 0.5)

    print("  [4/8] Closing hands around box...")
    smooth_move(model, data, viewer, q_close(), 0.8)
    hold(model, data, viewer, q_close(), 0.3)
    enable_weld(model, data)          # <-- BOX ATTACHES HERE
    hold(model, data, viewer, q_close(), 0.3)

    print("  [5/8] Lifting box...")
    smooth_move(model, data, viewer, q_lift(), 2.0)
    hold(model, data, viewer, q_lift(), 0.8)

    print("  [6/8] Turning right...")
    smooth_move(model, data, viewer, q_turn(), 2.5)
    hold(model, data, viewer, q_turn(), 0.8)

    print("  [7/8] Placing box...")
    smooth_move(model, data, viewer, q_place(), 1.5)
    hold(model, data, viewer, q_place(), 0.5)
    disable_weld(model, data)         # <-- BOX RELEASES HERE
    hold(model, data, viewer, q_place(), 0.5)

    print("  [8/8] Returning to stand...")
    smooth_move(model, data, viewer, q_stand(), 2.5)
    hold(model, data, viewer, q_stand(), 1.0)

    print("\n  ✓ Pick & place complete!")

def main():
    print("=" * 50)
    print("  G1 Humanoid — Pick & Place (Fixed Base)")
    print("=" * 50)

    try:
        model = load_model()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print(f"Loaded — nu={model.nu} nq={model.nq}")

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    q0 = q_stand()
    data.qpos[0:29] = q0
    mujoco.mj_forward(model, data)

    print("Settling...")
    for _ in range(3000):
        data.ctrl[:] = pd_torque(data, q0)
        mujoco.mj_step(model, data)

    print("Opening viewer...\n")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        hold(model, data, viewer, q0, 1.0)

        loop = 1
        while viewer.is_running():
            print(f"\n── Loop {loop} ──")
            run_sequence(model, data, viewer)
            loop += 1

        print("Close viewer to exit.")

if __name__ == "__main__":
    main()
