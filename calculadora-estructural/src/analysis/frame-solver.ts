export type SupportType = "fixed" | "pinned";

export interface ThreeStoryInput {
  type: "three-story";
  E: number;
  A: number;
  span: number;
  rightOverhang: number;
  tipLoads: [number, number, number];
  storyHeights: [number, number, number];
  columnInertias: [number, number, number];
  beamInertia: number;
  leftSupport: SupportType;
  rightSupport: SupportType;
  floorLoads: [number, number, number];
  lateralLoads: [number, number, number];
}

export interface InclinedFrameInput {
  type: "inclined";
  E: number;
  A: number;
  ABheight: number;
  BCspan: number;
  CDverticalProjection: number;
  angleDegrees: number;
  inertias: { AB: number; BC: number; CD: number };
  supportA: SupportType;
  supportD: SupportType;
  distributedLoad: number;
  pointLoadAtB: number;
}

export type FrameInput = ThreeStoryInput | InclinedFrameInput;

export interface FrameNode {
  id: string;
  x: number;
  y: number;
}

export interface FrameMemberDefinition {
  id: string;
  startNode: string;
  endNode: string;
  E: number;
  A: number;
  I: number;
  /** Signed vertical load in global coordinates; positive acts downward. */
  distributedLoad?: number;
  /** Signed global point loads applied inside the member. */
  pointLoads?: Array<{
    x: number;
    fx: number;
    fy: number;
  }>;
}

export interface FrameSupport {
  nodeId: string;
  ux: boolean;
  uy: boolean;
  rotation: boolean;
}

export interface NodalLoad {
  nodeId: string;
  fx?: number;
  fy?: number;
  moment?: number;
}

export interface DriftLevel {
  id: string;
  height: number;
  lowerNodeIds: string[];
  upperNodeIds: string[];
}

export interface FrameDefinition {
  label: string;
  nodes: FrameNode[];
  members: FrameMemberDefinition[];
  supports: FrameSupport[];
  nodalLoads?: NodalLoad[];
  driftLevels?: DriftLevel[];
}

export interface DiagramPoint {
  x: number;
  axial: number;
  shear: number;
  moment: number;
}

export interface MemberResult {
  id: string;
  length: number;
  angleDegrees: number;
  start: { axial: number; shear: number; moment: number };
  end: { axial: number; shear: number; moment: number };
  diagram: DiagramPoint[];
}

export interface ReactionResult {
  nodeId: string;
  fx: number;
  fy: number;
  moment: number;
}

export interface NodeDisplacement {
  nodeId: string;
  ux: number;
  uy: number;
  rotation: number;
}

export interface StoryDrift {
  id: string;
  horizontalDisplacement: number;
  driftRatio: number;
}

export interface FrameResult {
  label: string;
  memberResults: MemberResult[];
  reactions: ReactionResult[];
  displacements: NodeDisplacement[];
  storyDrifts: StoryDrift[];
  equilibrium: {
    sumFx: number;
    sumFy: number;
    sumMoment: number;
    normalizedForceResidual: number;
    normalizedMomentResidual: number;
  };
}

interface ElementRuntime {
  definition: FrameMemberDefinition;
  start: FrameNode;
  end: FrameNode;
  length: number;
  c: number;
  s: number;
  dofs: number[];
  transform: number[][];
  localStiffness: number[][];
  equivalentLocalLoads: number[];
}

const DOFS_PER_NODE = 3;
const DIAGRAM_DIVISIONS = 40;
const EPSILON = 1e-10;

export function makeDefaultThreeStoryInput(): ThreeStoryInput {
  return {
    type: "three-story",
    E: 2_000_000,
    A: 0.04,
    span: 5,
    rightOverhang: 1.5,
    tipLoads: [2, 2, 2],
    storyHeights: [4, 3, 3],
    columnInertias: [4, 3, 2],
    beamInertia: 1,
    leftSupport: "fixed",
    rightSupport: "fixed",
    floorLoads: [3, 3, 2],
    lateralLoads: [0, 0, 0],
  };
}

export function makeDefaultInclinedFrameInput(): InclinedFrameInput {
  return {
    type: "inclined",
    E: 2_000_000,
    A: 0.04,
    ABheight: 3,
    BCspan: 3.5,
    CDverticalProjection: 2,
    angleDegrees: 40,
    inertias: { AB: 1, BC: 1, CD: 1 },
    supportA: "fixed",
    supportD: "fixed",
    distributedLoad: 2.5,
    pointLoadAtB: 3,
  };
}

/**
 * Solve a first-order, linear-elastic, 2D Euler-Bernoulli frame using the
 * direct-stiffness method. Loads and reactions are global; member diagrams use
 * local member axes. N is positive in tension, V is positive in local +y, and
 * the plotted M sign is positive for sagging.
 */
export function solveFrame(input: FrameInput): FrameResult {
  return solveFrameModel(buildFrameDefinition(input));
}

/** Public low-level entry point used for numerical reference tests. */
export function solveFrameModel(definition: FrameDefinition): FrameResult {
  validateDefinition(definition);

  const nodeIndex = new Map(definition.nodes.map((node, index) => [node.id, index]));
  const dofCount = definition.nodes.length * DOFS_PER_NODE;
  const stiffness = zeroMatrix(dofCount, dofCount);
  const externalLoads = new Array<number>(dofCount).fill(0);
  const supports = new Map<string, FrameSupport>();

  for (const support of definition.supports) {
    supports.set(support.nodeId, support);
  }
  for (const load of definition.nodalLoads ?? []) {
    const base = nodeIndex.get(load.nodeId)! * DOFS_PER_NODE;
    externalLoads[base] += load.fx ?? 0;
    externalLoads[base + 1] += load.fy ?? 0;
    externalLoads[base + 2] += load.moment ?? 0;
  }

  const elements = definition.members.map((member) => {
    const start = definition.nodes[nodeIndex.get(member.startNode)!]!;
    const end = definition.nodes[nodeIndex.get(member.endNode)!]!;
    const element = createElementRuntime(member, start, end, nodeIndex);
    const globalStiffness = transformMatrix(element.localStiffness, element.transform);
    addMatrixAt(stiffness, globalStiffness, element.dofs);
    const globalLoads = multiplyTransposeVector(
      element.transform,
      element.equivalentLocalLoads,
    );
    addVectorAt(externalLoads, globalLoads, element.dofs);
    return element;
  });

  const restrainedDofs = new Set<number>();
  for (const support of definition.supports) {
    const base = nodeIndex.get(support.nodeId)! * DOFS_PER_NODE;
    if (support.ux) restrainedDofs.add(base);
    if (support.uy) restrainedDofs.add(base + 1);
    if (support.rotation) restrainedDofs.add(base + 2);
  }

  const freeDofs = Array.from({ length: dofCount }, (_, index) => index).filter(
    (index) => !restrainedDofs.has(index),
  );
  const displacements = new Array<number>(dofCount).fill(0);
  if (freeDofs.length > 0) {
    const reducedStiffness = freeDofs.map((row) =>
      freeDofs.map((column) => stiffness[row]![column]!),
    );
    const reducedLoads = freeDofs.map((dof) => externalLoads[dof]!);
    const freeDisplacements = solveLinearSystem(reducedStiffness, reducedLoads);
    freeDofs.forEach((dof, index) => {
      displacements[dof] = freeDisplacements[index]!;
    });
  }

  const residual = multiplyMatrixVector(stiffness, displacements).map(
    (value, index) => value - externalLoads[index]!,
  );
  const reactions = definition.supports.map((support) => {
    const base = nodeIndex.get(support.nodeId)! * DOFS_PER_NODE;
    return {
      nodeId: support.nodeId,
      fx: support.ux ? residual[base]! : 0,
      fy: support.uy ? residual[base + 1]! : 0,
      moment: support.rotation ? residual[base + 2]! : 0,
    };
  });

  const memberResults = elements.map((element) =>
    calculateMemberResult(element, displacements),
  );
  const nodeDisplacements = definition.nodes.map((node, index) => ({
    nodeId: node.id,
    ux: displacements[index * DOFS_PER_NODE]!,
    uy: displacements[index * DOFS_PER_NODE + 1]!,
    rotation: displacements[index * DOFS_PER_NODE + 2]!,
  }));

  return {
    label: definition.label,
    memberResults,
    reactions,
    displacements: nodeDisplacements,
    storyDrifts: calculateStoryDrifts(
      definition.driftLevels ?? [],
      nodeDisplacements,
    ),
    equilibrium: calculateEquilibrium(definition.nodes, externalLoads, reactions),
  };
}

function buildFrameDefinition(input: FrameInput): FrameDefinition {
  validateUserInput(input);

  if (input.type === "three-story") {
    return buildThreeStoryDefinition(input);
  }
  return buildInclinedDefinition(input);
}

function buildThreeStoryDefinition(input: ThreeStoryInput): FrameDefinition {
  const [h1, h2, h3] = input.storyHeights;
  const [I1, I2, I3] = input.columnInertias;
  const y1 = h1;
  const y2 = h1 + h2;
  const y3 = h1 + h2 + h3;
  const nodes: FrameNode[] = [
    { id: "A", x: 0, y: 0 },
    { id: "B", x: 0, y: y1 },
    { id: "C", x: 0, y: y2 },
    { id: "D", x: 0, y: y3 },
    { id: "H", x: input.span, y: 0 },
    { id: "G", x: input.span, y: y1 },
    { id: "F", x: input.span, y: y2 },
    { id: "E", x: input.span, y: y3 },
    { id: "T1", x: input.span + input.rightOverhang, y: y1 },
    { id: "T2", x: input.span + input.rightOverhang, y: y2 },
    { id: "T3", x: input.span + input.rightOverhang, y: y3 },
  ];
  const members: FrameMemberDefinition[] = [
    member("AB", "A", "B", input.E, input.A, I1),
    member("BC", "B", "C", input.E, input.A, I2),
    member("CD", "C", "D", input.E, input.A, I3),
    member("HG", "H", "G", input.E, input.A, I1),
    member("GF", "G", "F", input.E, input.A, I2),
    member("FE", "F", "E", input.E, input.A, I3),
    member("BG", "B", "G", input.E, input.A, input.beamInertia, input.floorLoads[0]),
    member("CF", "C", "F", input.E, input.A, input.beamInertia, input.floorLoads[1]),
    member("DE", "D", "E", input.E, input.A, input.beamInertia, input.floorLoads[2]),
    member("GT1", "G", "T1", input.E, input.A, input.beamInertia, input.floorLoads[0]),
    member("FT2", "F", "T2", input.E, input.A, input.beamInertia, input.floorLoads[1]),
    member("ET3", "E", "T3", input.E, input.A, input.beamInertia, input.floorLoads[2]),
  ];
  const nodesByLevel = [
    ["B", "G"],
    ["C", "F"],
    ["D", "E"],
  ];
  const nodalLoads: NodalLoad[] = [
    ...nodesByLevel.flatMap((ids, index) =>
      ids.map((nodeId) => ({
        nodeId,
        fx: input.lateralLoads[index] / 2,
      })),
    ),
    ...(["T1", "T2", "T3"] as const).map((nodeId, index) => ({
      nodeId,
      fy: -input.tipLoads[index],
    })),
  ];

  return {
    label: "Marco de tres niveles",
    nodes,
    members,
    supports: [
      baseSupport("A", input.leftSupport),
      baseSupport("H", input.rightSupport),
    ],
    nodalLoads,
    driftLevels: [
      {
        id: "Nivel 1",
        height: h1,
        lowerNodeIds: ["A", "H"],
        upperNodeIds: ["B", "G"],
      },
      {
        id: "Nivel 2",
        height: h2,
        lowerNodeIds: ["B", "G"],
        upperNodeIds: ["C", "F"],
      },
      {
        id: "Nivel 3",
        height: h3,
        lowerNodeIds: ["C", "F"],
        upperNodeIds: ["D", "E"],
      },
    ],
  };
}

function buildInclinedDefinition(input: InclinedFrameInput): FrameDefinition {
  const radians = (input.angleDegrees * Math.PI) / 180;
  const dx = input.CDverticalProjection * Math.tan(radians);
  const nodes: FrameNode[] = [
    { id: "A", x: 0, y: 0 },
    { id: "B", x: 0, y: input.ABheight },
    { id: "C", x: input.BCspan, y: input.ABheight },
    {
      id: "D",
      x: input.BCspan + dx,
      y: input.ABheight - input.CDverticalProjection,
    },
  ];

  return {
    label: "Marco inclinado",
    nodes,
    members: [
      member("AB", "A", "B", input.E, input.A, input.inertias.AB),
      member(
        "BC",
        "B",
        "C",
        input.E,
        input.A,
        input.inertias.BC,
        input.distributedLoad,
      ),
      member("CD", "C", "D", input.E, input.A, input.inertias.CD),
    ],
    supports: [
      baseSupport("A", input.supportA),
      baseSupport("D", input.supportD),
    ],
    nodalLoads: [{ nodeId: "B", fy: -input.pointLoadAtB }],
  };
}

function member(
  id: string,
  startNode: string,
  endNode: string,
  E: number,
  A: number,
  I: number,
  distributedLoad = 0,
): FrameMemberDefinition {
  return { id, startNode, endNode, E, A, I, distributedLoad };
}

function baseSupport(nodeId: string, type: SupportType): FrameSupport {
  return {
    nodeId,
    ux: true,
    uy: true,
    rotation: type === "fixed",
  };
}

function createElementRuntime(
  definition: FrameMemberDefinition,
  start: FrameNode,
  end: FrameNode,
  nodeIndex: Map<string, number>,
): ElementRuntime {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy);
  const c = dx / length;
  const s = dy / length;
  const transform = zeroMatrix(6, 6);
  for (const offset of [0, 3]) {
    transform[offset]![offset] = c;
    transform[offset]![offset + 1] = s;
    transform[offset + 1]![offset] = -s;
    transform[offset + 1]![offset + 1] = c;
    transform[offset + 2]![offset + 2] = 1;
  }

  return {
    definition,
    start,
    end,
    length,
    c,
    s,
    dofs: [
      nodeIndex.get(start.id)! * DOFS_PER_NODE,
      nodeIndex.get(start.id)! * DOFS_PER_NODE + 1,
      nodeIndex.get(start.id)! * DOFS_PER_NODE + 2,
      nodeIndex.get(end.id)! * DOFS_PER_NODE,
      nodeIndex.get(end.id)! * DOFS_PER_NODE + 1,
      nodeIndex.get(end.id)! * DOFS_PER_NODE + 2,
    ],
    transform,
    localStiffness: localFrameStiffness(
      definition.E,
      definition.A,
      definition.I,
      length,
    ),
    equivalentLocalLoads: equivalentElementLoads(definition, length, c, s),
  };
}

function localFrameStiffness(E: number, A: number, I: number, L: number): number[][] {
  const k = zeroMatrix(6, 6);
  const axial = (E * A) / L;
  k[0]![0] = axial;
  k[0]![3] = -axial;
  k[3]![0] = -axial;
  k[3]![3] = axial;

  const a = (12 * E * I) / L ** 3;
  const b = (6 * E * I) / L ** 2;
  const d = (4 * E * I) / L;
  const e = (2 * E * I) / L;
  const flexural = [
    [1, 1, a],
    [1, 2, b],
    [1, 4, -a],
    [1, 5, b],
    [2, 1, b],
    [2, 2, d],
    [2, 4, -b],
    [2, 5, e],
    [4, 1, -a],
    [4, 2, -b],
    [4, 4, a],
    [4, 5, -b],
    [5, 1, b],
    [5, 2, e],
    [5, 4, -b],
    [5, 5, d],
  ] as const;
  for (const [row, column, value] of flexural) {
    k[row]![column] = value;
  }
  return k;
}

function equivalentElementLoads(
  member: FrameMemberDefinition,
  L: number,
  c: number,
  s: number,
): number[] {
  const loads = new Array<number>(6).fill(0);
  const verticalLoad = member.distributedLoad ?? 0;
  const qx = -verticalLoad * s;
  const qy = -verticalLoad * c;
  loads[0] += (qx * L) / 2;
  loads[3] += (qx * L) / 2;
  loads[1] += (qy * L) / 2;
  loads[2] += (qy * L ** 2) / 12;
  loads[4] += (qy * L) / 2;
  loads[5] -= (qy * L ** 2) / 12;

  for (const point of member.pointLoads ?? []) {
    const r = point.x / L;
    const localFx = c * point.fx + s * point.fy;
    const localFy = -s * point.fx + c * point.fy;
    loads[0] += localFx * (1 - r);
    loads[3] += localFx * r;
    const n1 = 1 - 3 * r ** 2 + 2 * r ** 3;
    const n2 = L * (r - 2 * r ** 2 + r ** 3);
    const n3 = 3 * r ** 2 - 2 * r ** 3;
    const n4 = L * (-(r ** 2) + r ** 3);
    loads[1] += localFy * n1;
    loads[2] += localFy * n2;
    loads[4] += localFy * n3;
    loads[5] += localFy * n4;
  }
  return loads;
}

function calculateMemberResult(
  element: ElementRuntime,
  globalDisplacements: number[],
): MemberResult {
  const globalElementDisplacements = element.dofs.map(
    (dof) => globalDisplacements[dof]!,
  );
  const localDisplacements = multiplyMatrixVector(
    element.transform,
    globalElementDisplacements,
  );
  const localActions = multiplyMatrixVector(
    element.localStiffness,
    localDisplacements,
  ).map((value, index) => value - element.equivalentLocalLoads[index]!);

  const verticalLoad = element.definition.distributedLoad ?? 0;
  const qx = -verticalLoad * element.s;
  const qy = -verticalLoad * element.c;
  const pointLoads = (element.definition.pointLoads ?? []).map((point) => ({
    ...point,
    localFx: element.c * point.fx + element.s * point.fy,
    localFy: -element.s * point.fx + element.c * point.fy,
  }));
  const stations = new Set<number>();
  for (let index = 0; index <= DIAGRAM_DIVISIONS; index += 1) {
    stations.add((element.length * index) / DIAGRAM_DIVISIONS);
  }
  for (const point of pointLoads) stations.add(point.x);

  const sortedStations = [...stations].sort((a, b) => a - b);
  const diagram: DiagramPoint[] = [];
  for (const x of sortedStations) {
    const atLoad = pointLoads.some((point) => Math.abs(point.x - x) < EPSILON);
    const before = diagramValuesAt(x, false, localActions, qx, qy, pointLoads);
    if (atLoad) diagram.push({ x, ...before });
    const after = diagramValuesAt(x, true, localActions, qx, qy, pointLoads);
    diagram.push({ x, ...after });
  }

  return {
    id: element.definition.id,
    length: element.length,
    angleDegrees: (Math.atan2(element.s, element.c) * 180) / Math.PI,
    start: {
      axial: localActions[0]!,
      shear: localActions[1]!,
      moment: localActions[2]!,
    },
    end: {
      axial: localActions[3]!,
      shear: localActions[4]!,
      moment: localActions[5]!,
    },
    diagram,
  };
}

function diagramValuesAt(
  x: number,
  includeLoadsAtStation: boolean,
  actions: number[],
  qx: number,
  qy: number,
  pointLoads: Array<{
    x: number;
    localFx: number;
    localFy: number;
  }>,
): Omit<DiagramPoint, "x"> {
  let axial = -actions[0]! - qx * x;
  let shear = actions[1]! + qy * x;
  let moment = -actions[2]! + actions[1]! * x + (qy * x ** 2) / 2;
  for (const point of pointLoads) {
    const isPastLoad =
      point.x < x - EPSILON ||
      (includeLoadsAtStation && Math.abs(point.x - x) < EPSILON);
    if (!isPastLoad) continue;
    axial -= point.localFx;
    shear += point.localFy;
    moment += point.localFy * (x - point.x);
  }
  return { axial, shear, moment };
}

function calculateStoryDrifts(
  levels: DriftLevel[],
  displacements: NodeDisplacement[],
): StoryDrift[] {
  const byId = new Map(displacements.map((item) => [item.nodeId, item]));
  return levels.map((level) => {
    const lower = average(level.lowerNodeIds.map((id) => byId.get(id)!.ux));
    const upper = average(level.upperNodeIds.map((id) => byId.get(id)!.ux));
    const displacement = upper - lower;
    return {
      id: level.id,
      horizontalDisplacement: displacement,
      driftRatio: level.height > EPSILON ? displacement / level.height : 0,
    };
  });
}

function calculateEquilibrium(
  nodes: FrameNode[],
  externalLoads: number[],
  reactions: ReactionResult[],
): FrameResult["equilibrium"] {
  let sumFx = 0;
  let sumFy = 0;
  let sumMoment = 0;
  let totalAppliedForce = 0;
  let totalAppliedMoment = 0;
  let totalReactionForce = 0;
  let totalReactionMoment = 0;
  nodes.forEach((node, index) => {
    const fx = externalLoads[index * DOFS_PER_NODE]!;
    const fy = externalLoads[index * DOFS_PER_NODE + 1]!;
    const moment = externalLoads[index * DOFS_PER_NODE + 2]!;
    sumFx += fx;
    sumFy += fy;
    sumMoment += moment + node.x * fy - node.y * fx;
    totalAppliedForce += Math.hypot(fx, fy);
    totalAppliedMoment += Math.abs(moment + node.x * fy - node.y * fx);
  });
  for (const reaction of reactions) {
    const node = nodes.find((candidate) => candidate.id === reaction.nodeId)!;
    sumFx += reaction.fx;
    sumFy += reaction.fy;
    sumMoment += reaction.moment + node.x * reaction.fy - node.y * reaction.fx;
    totalReactionForce += Math.hypot(reaction.fx, reaction.fy);
    totalReactionMoment += Math.abs(
      reaction.moment + node.x * reaction.fy - node.y * reaction.fx,
    );
  }
  return {
    sumFx,
    sumFy,
    sumMoment,
    normalizedForceResidual:
      Math.hypot(sumFx, sumFy) /
      Math.max(totalAppliedForce + totalReactionForce, 1),
    normalizedMomentResidual:
      Math.abs(sumMoment) /
      Math.max(totalAppliedMoment + totalReactionMoment, 1),
  };
}

function validateUserInput(input: FrameInput): void {
  assertPositive(input.E, "E");
  assertPositive(input.A, "A");
  if (input.type === "three-story") {
    assertPositive(input.span, "La luz del pórtico");
    assertPositive(input.rightOverhang, "La longitud del voladizo derecho");
    input.storyHeights.forEach((value, index) =>
      assertPositive(value, `Altura del nivel ${index + 1}`),
    );
    input.columnInertias.forEach((value, index) =>
      assertPositive(value, `Inercia de columna del nivel ${index + 1}`),
    );
    assertPositive(input.beamInertia, "Inercia de viga");
    validateSupportType(input.leftSupport, "Apoyo izquierdo");
    validateSupportType(input.rightSupport, "Apoyo derecho");
    input.floorLoads.forEach((value, index) =>
      assertFinite(value, `Carga distribuida del nivel ${index + 1}`),
    );
    input.lateralLoads.forEach((value, index) =>
      assertFinite(value, `Carga lateral del nivel ${index + 1}`),
    );
    input.tipLoads.forEach((value, index) => {
      assertFinite(value, `Carga puntual en la punta del voladizo del nivel ${index + 1}`);
      if (value < 0) {
        throw new Error(
          `La carga puntual descendente en la punta del voladizo del nivel ${index + 1} debe ser cero o positiva.`,
        );
      }
    });
  } else if (input.type === "inclined") {
    assertPositive(input.ABheight, "Altura AB");
    assertPositive(input.BCspan, "Luz BC");
    assertPositive(input.CDverticalProjection, "Proyección vertical CD");
    if (input.CDverticalProjection >= input.ABheight) {
      throw new Error(
        "La proyección vertical CD debe ser menor que la altura AB para mantener el apoyo D por encima de A.",
      );
    }
    assertFinite(input.angleDegrees, "Ángulo CD");
    if (input.angleDegrees < 10 || input.angleDegrees > 80) {
      throw new Error("El ángulo CD debe estar entre 10° y 80° respecto de la vertical.");
    }
    assertPositive(input.inertias.AB, "Inercia AB");
    assertPositive(input.inertias.BC, "Inercia BC");
    assertPositive(input.inertias.CD, "Inercia CD");
    validateSupportType(input.supportA, "Apoyo A");
    validateSupportType(input.supportD, "Apoyo D");
    assertFinite(input.distributedLoad, "Carga distribuida BC");
    assertFinite(input.pointLoadAtB, "Carga puntual en B");
    if (input.pointLoadAtB < 0) {
      throw new Error("La carga puntual descendente en B debe ser cero o positiva.");
    }
  } else {
    throw new Error("El tipo de estructura seleccionado no está soportado.");
  }
}

function validateDefinition(definition: FrameDefinition): void {
  if (definition.nodes.length < 2) {
    throw new Error("El modelo debe incluir al menos dos nudos.");
  }
  const nodeIds = new Set<string>();
  for (const node of definition.nodes) {
    if (!node.id || nodeIds.has(node.id)) {
      throw new Error("Los identificadores de nudo deben ser únicos y no vacíos.");
    }
    nodeIds.add(node.id);
    assertFinite(node.x, `Coordenada X del nudo ${node.id}`);
    assertFinite(node.y, `Coordenada Y del nudo ${node.id}`);
  }
  if (definition.members.length === 0) {
    throw new Error("El modelo debe incluir al menos una barra.");
  }
  const memberIds = new Set<string>();
  for (const member of definition.members) {
    if (!member.id || memberIds.has(member.id)) {
      throw new Error("Los identificadores de barra deben ser únicos y no vacíos.");
    }
    memberIds.add(member.id);
    if (!nodeIds.has(member.startNode) || !nodeIds.has(member.endNode)) {
      throw new Error(`La barra ${member.id} referencia un nudo inexistente.`);
    }
    if (member.startNode === member.endNode) {
      throw new Error(`La barra ${member.id} no puede iniciar y terminar en el mismo nudo.`);
    }
    assertPositive(member.E, `E de la barra ${member.id}`);
    assertPositive(member.A, `Área de la barra ${member.id}`);
    assertPositive(member.I, `Inercia de la barra ${member.id}`);
    assertFinite(member.distributedLoad ?? 0, `Carga distribuida de la barra ${member.id}`);
    for (const point of member.pointLoads ?? []) {
      assertFinite(point.x, `Posición de carga puntual en ${member.id}`);
      assertFinite(point.fx, `Componente horizontal de carga puntual en ${member.id}`);
      assertFinite(point.fy, `Componente vertical de carga puntual en ${member.id}`);
      const start = definition.nodes.find((node) => node.id === member.startNode)!;
      const end = definition.nodes.find((node) => node.id === member.endNode)!;
      const length = Math.hypot(end.x - start.x, end.y - start.y);
      if (point.x < 0 || point.x > length) {
        throw new Error(`La carga puntual de ${member.id} debe ubicarse sobre la barra.`);
      }
    }
  }
  if (definition.supports.length === 0) {
    throw new Error("El modelo debe incluir al menos un apoyo.");
  }
  const supportNodes = new Set<string>();
  for (const support of definition.supports) {
    if (!nodeIds.has(support.nodeId)) {
      throw new Error(`El apoyo referencia el nudo inexistente ${support.nodeId}.`);
    }
    if (supportNodes.has(support.nodeId)) {
      throw new Error(`El nudo ${support.nodeId} tiene apoyos duplicados.`);
    }
    supportNodes.add(support.nodeId);
  }
  for (const load of definition.nodalLoads ?? []) {
    if (!nodeIds.has(load.nodeId)) {
      throw new Error(`La carga puntual referencia el nudo inexistente ${load.nodeId}.`);
    }
    assertFinite(load.fx ?? 0, `Carga horizontal en ${load.nodeId}`);
    assertFinite(load.fy ?? 0, `Carga vertical en ${load.nodeId}`);
    assertFinite(load.moment ?? 0, `Momento aplicado en ${load.nodeId}`);
  }
}

function validateSupportType(value: SupportType, name: string): void {
  if (value !== "fixed" && value !== "pinned") {
    throw new Error(`${name}: seleccione un apoyo fijo o articulado.`);
  }
}

function assertPositive(value: number, name: string): void {
  assertFinite(value, name);
  if (value <= 0) {
    throw new Error(`${name} debe ser mayor que cero.`);
  }
}

function assertFinite(value: number, name: string): void {
  if (!Number.isFinite(value)) {
    throw new Error(`${name} debe ser un número finito.`);
  }
}

function solveLinearSystem(matrix: number[][], vector: number[]): number[] {
  const n = vector.length;
  const a = matrix.map((row, index) => [...row, vector[index]!]);
  let scale = 0;
  for (const row of matrix) {
    for (const value of row) scale = Math.max(scale, Math.abs(value));
  }
  if (!Number.isFinite(scale) || scale === 0) {
    throw new Error("La estructura es inestable: la matriz de rigidez está vacía.");
  }

  for (let column = 0; column < n; column += 1) {
    let pivotRow = column;
    for (let row = column + 1; row < n; row += 1) {
      if (Math.abs(a[row]![column]!) > Math.abs(a[pivotRow]![column]!)) {
        pivotRow = row;
      }
    }
    if (Math.abs(a[pivotRow]![column]!) < scale * 1e-13) {
      throw new Error(
        "La estructura es inestable o está mal restringida. Revise conectividad, geometría y apoyos.",
      );
    }
    [a[column], a[pivotRow]] = [a[pivotRow]!, a[column]!];
    const pivot = a[column]![column]!;
    for (let j = column; j <= n; j += 1) {
      a[column]![j] /= pivot;
    }
    for (let row = 0; row < n; row += 1) {
      if (row === column) continue;
      const factor = a[row]![column]!;
      if (factor === 0) continue;
      for (let j = column; j <= n; j += 1) {
        a[row]![j] -= factor * a[column]![j]!;
      }
    }
  }
  const result = a.map((row) => row[n]!);
  if (result.some((value) => !Number.isFinite(value))) {
    throw new Error("No se pudo obtener un resultado finito; revise la escala de los datos.");
  }
  return result;
}

function transformMatrix(local: number[][], transform: number[][]): number[][] {
  const transposeTimesLocal = zeroMatrix(6, 6);
  for (let row = 0; row < 6; row += 1) {
    for (let column = 0; column < 6; column += 1) {
      for (let k = 0; k < 6; k += 1) {
        transposeTimesLocal[row]![column] +=
          transform[k]![row]! * local[k]![column]!;
      }
    }
  }
  const global = zeroMatrix(6, 6);
  for (let row = 0; row < 6; row += 1) {
    for (let column = 0; column < 6; column += 1) {
      for (let k = 0; k < 6; k += 1) {
        global[row]![column] +=
          transposeTimesLocal[row]![k]! * transform[k]![column]!;
      }
    }
  }
  return global;
}

function multiplyTransposeVector(matrix: number[][], vector: number[]): number[] {
  return matrix[0]!.map((_, column) =>
    matrix.reduce((sum, row, index) => sum + row[column]! * vector[index]!, 0),
  );
}

function multiplyMatrixVector(matrix: number[][], vector: number[]): number[] {
  return matrix.map((row) =>
    row.reduce((sum, value, index) => sum + value * vector[index]!, 0),
  );
}

function addMatrixAt(target: number[][], values: number[][], indices: number[]): void {
  for (let row = 0; row < indices.length; row += 1) {
    for (let column = 0; column < indices.length; column += 1) {
      target[indices[row]!]![indices[column]!] += values[row]![column]!;
    }
  }
}

function addVectorAt(target: number[], values: number[], indices: number[]): void {
  indices.forEach((index, position) => {
    target[index] += values[position]!;
  });
}

function zeroMatrix(rows: number, columns: number): number[][] {
  return Array.from({ length: rows }, () => new Array<number>(columns).fill(0));
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
