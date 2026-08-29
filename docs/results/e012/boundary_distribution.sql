-- 从 docs/results/e012/boundary_distribution.json 复现报告中的 paired median 与 bootstrap 区间。
WITH paired(boundary, metric, median_difference, ci95_low, ci95_high, unit, n) AS (
  VALUES
    ('Reach→Grasp', 'TCP-object XY error', 0.02599069317083326, 0.018819617606476536, 0.03110429356811795, 'm', 31),
    ('Reach→Grasp', 'TCP-object relative Z', -0.011440223082900047, -0.017976203933358192, -0.006394876167178154, 'm', 31),
    ('Reach→Grasp', 'Gripper opening', -0.01841270923614502, -0.03315389156341553, -0.013126611709594727, 'opening ratio', 31),
    ('Reach→Grasp', 'Arm mean-pairwise disagreement', 0.03186699375510216, 0.028929969295859337, 0.040663134306669235, 'normalized action', 31),
    ('Reach→Grasp', 'Gripper mean-pairwise disagreement', 0.02442866563796997, 0.018256237730383873, 0.028356969356536865, 'normalized action', 31),
    ('Reach→Grasp', 'Boundary arrival delay', 44.0, 38.0, 52.0, 'control steps', 31),
    ('Grasp→Lift', 'Object angular speed', 4.853168585403354, 1.250443070265686, 6.917546624046038, 'rad/s', 10),
    ('Grasp→Lift', 'Joint velocity RMS', 0.005303063662722707, 0.0032124354038387537, 0.009476952604018152, 'rad/s', 10),
    ('Grasp→Lift', 'Gripper opening', 0.06686671078205109, 0.04001152515411377, 0.11242949962615967, 'opening ratio', 10),
    ('Grasp→Lift', 'Arm mean-pairwise disagreement', 0.029620669782161713, 0.022762715816497803, 0.036626920104026794, 'normalized action', 10),
    ('Grasp→Lift', 'Gripper mean-pairwise disagreement', 1.2597848773002625, 0.9920904636383057, 1.316806674003601, 'normalized action', 10),
    ('Grasp→Lift', 'Boundary arrival delay', 51.0, 27.5, 78.5, 'control steps', 10)
)
SELECT * FROM paired;
