-- 从 docs/results/e012/collection_manifest.json 复现报告中的冻结身份表。
WITH identity(asset, identity, status) AS (
  VALUES
    ('Collection source tree', 'source-tree-sha256:a847e9f90fb255405714351379b1530691f4224f6a29a8e21daad76a5ef8ee00', 'MATCH'),
    ('E011 Layer 12 checkpoint', 'a542076f291e29b68e3d28930b15c40396d511a44eb358c2eaeb4e113c041ad6', 'MATCH'),
    ('Frozen D0', 'bc024b6b9c566ca9500945fb6ac262bf657bee713d8a5816229bdc8478139407', 'MATCH'),
    ('Qwen revision', '15852e8c16360a2fea060d615a32b45270f8a8fc', 'MATCH'),
    ('Formal environment seeds', 'RG 30000..30099; GL 30100..30199; overlap=[]', 'MATCH')
)
SELECT * FROM identity;
