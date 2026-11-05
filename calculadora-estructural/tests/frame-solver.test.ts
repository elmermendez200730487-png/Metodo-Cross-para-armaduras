import assert from "node:assert/strict";
import test from "node:test";

import {
  makeDefaultInclinedFrameInput,
  makeDefaultThreeStoryInput,
  solveFrame,
  solveFrameModel,
  type FrameDefinition,
} from "../src/analysis/frame-solver";

function near(actual: number, expected: number, tolerance = 1e-7): void {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `Expected ${actual} to be within ${tolerance} of ${expected}`,
  );
}

function simpleBeam(load: {
  distributedLoad?: number;
  pointLoads?: Array<{ x: number; fx: number; fy: number }>;
  supports?: FrameDefinition["supports"];
}): FrameDefinition {
  return {
    label: "Reference beam",
    nodes: [
      { id: "L", x: 0, y: 0 },
      { id: "R", x: 5, y: 0 },
    ],
    members: [
      {
        id: "LR",
        startNode: "L",
        endNode: "R",
        E: 1,
        A: 1,
        I: 1,
        distributedLoad: load.distributedLoad ?? 0,
        pointLoads: load.pointLoads,
      },
    ],
    supports:
      load.supports ??
      [
        { nodeId: "L", ux: true, uy: true, rotation: false },
        { nodeId: "R", ux: true, uy: true, rotation: false },
      ],
  };
}

test("simply supported beam under uniform load has the reference reactions and moment", () => {
  const result = solveFrameModel(simpleBeam({ distributedLoad: 3 }));
  const left = result.reactions.find((reaction) => reaction.nodeId === "L")!;
  const right = result.reactions.find((reaction) => reaction.nodeId === "R")!;
  const diagram = result.memberResults[0]!.diagram;
  const midspan = diagram.find((point) => Math.abs(point.x - 2.5) < 1e-9)!;

  near(left.fy, 7.5);
  near(right.fy, 7.5);
  near(left.moment, 0);
  near(right.moment, 0);
  near(midspan.moment, 9.375);
  near(result.equilibrium.sumFy, 0);
  near(result.equilibrium.sumMoment, 0);
});

test("simply supported beam with a central point load has equal reactions and a shear jump", () => {
  const result = solveFrameModel(
    simpleBeam({
      pointLoads: [{ x: 2.5, fx: 0, fy: -10 }],
    }),
  );
  const left = result.reactions.find((reaction) => reaction.nodeId === "L")!;
  const right = result.reactions.find((reaction) => reaction.nodeId === "R")!;
  const atLoad = result.memberResults[0]!.diagram.filter(
    (point) => Math.abs(point.x - 2.5) < 1e-9,
  );

  near(left.fy, 5);
  near(right.fy, 5);
  assert.equal(atLoad.length, 2, "The diagram should show both sides of the point load.");
  near(atLoad[0]!.shear, 5);
  near(atLoad[1]!.shear, -5);
  near(atLoad[0]!.moment, 12.5);
  near(atLoad[1]!.moment, 12.5);
  near(result.equilibrium.sumFy, 0);
  near(result.equilibrium.sumMoment, 0);
});

test("fixed-end beam actions and the sagging-positive moment curve match uniform-load theory", () => {
  const result = solveFrameModel(
    simpleBeam({
      distributedLoad: 3,
      supports: [
        { nodeId: "L", ux: true, uy: true, rotation: true },
        { nodeId: "R", ux: true, uy: true, rotation: true },
      ],
    }),
  );
  const member = result.memberResults[0]!;
  const midspan = member.diagram.find((point) => Math.abs(point.x - 2.5) < 1e-9)!;

  near(member.start.moment, 6.25);
  near(member.end.moment, -6.25);
  near(member.diagram[0]!.moment, -6.25);
  near(midspan.moment, 3.125);
  near(result.equilibrium.sumFy, 0);
});

test("three-story rigid joints balance end moments and equilibrium", () => {
  const input = makeDefaultThreeStoryInput();
  input.lateralLoads = [4, 8, 12];
  const result = solveFrame(input);
  const jointMoments = new Map<string, number>();

  for (const member of result.memberResults) {
    const definition = [
      ["AB", "A", "B"],
      ["BC", "B", "C"],
      ["CD", "C", "D"],
      ["HG", "H", "G"],
      ["GF", "G", "F"],
      ["FE", "F", "E"],
      ["BG", "B", "G"],
      ["CF", "C", "F"],
      ["DE", "D", "E"],
      ["GT1", "G", "T1"],
      ["FT2", "F", "T2"],
      ["ET3", "E", "T3"],
    ].find(([id]) => id === member.id)!;
    jointMoments.set(
      definition[1]!,
      (jointMoments.get(definition[1]!) ?? 0) + member.start.moment,
    );
    jointMoments.set(
      definition[2]!,
      (jointMoments.get(definition[2]!) ?? 0) + member.end.moment,
    );
  }

  for (const nodeId of ["B", "C", "D", "G", "F", "E"]) {
    near(jointMoments.get(nodeId) ?? 0, 0, 1e-5);
  }
  assert.ok(result.equilibrium.normalizedForceResidual < 1e-10);
  assert.ok(result.equilibrium.normalizedMomentResidual < 1e-10);
});

test("lateral story load creates nonzero sidesway and balances at the supports", () => {
  const input = makeDefaultThreeStoryInput();
  input.floorLoads = [0, 0, 0];
  input.lateralLoads = [0, 0, 20];
  const result = solveFrame(input);
  const topAverageUx =
    (result.displacements.find((node) => node.nodeId === "D")!.ux +
      result.displacements.find((node) => node.nodeId === "E")!.ux) /
    2;
  const horizontalReaction = result.reactions.reduce(
    (sum, reaction) => sum + reaction.fx,
    0,
  );

  assert.ok(topAverageUx > 0, "A rightward lateral load should cause rightward sway.");
  assert.ok(result.storyDrifts.some((level) => level.horizontalDisplacement > 0));
  near(horizontalReaction, -20);
  near(result.equilibrium.sumFx, 0);
  near(result.equilibrium.sumMoment, 0);
});

test("symmetric gravity-only loading does not create frame sidesway", () => {
  const result = solveFrame(makeDefaultThreeStoryInput());
  const verticalReaction = result.reactions.reduce(
    (sum, reaction) => sum + reaction.fy,
    0,
  );

  for (const drift of result.storyDrifts) {
    near(drift.horizontalDisplacement, 0, 1e-9);
  }
  near(
    verticalReaction,
    (3 + 3 + 2) * (input.span + input.rightOverhang) +
      input.tipLoads.reduce((sum, load) => sum + load, 0),
  );
  near(result.equilibrium.sumFy, 0);
});

test("right overhangs receive the full distributed load and a downward tip load", () => {
  const input = makeDefaultThreeStoryInput();
  input.floorLoads = [0, 0, 2];
  input.tipLoads = [0, 0, 5];
  input.lateralLoads = [0, 0, 0];
  const result = solveFrame(input);
  const verticalReaction = result.reactions.reduce(
    (sum, reaction) => sum + reaction.fy,
    0,
  );
  const topOverhang = result.memberResults.find((member) => member.id === "ET3")!;

  near(verticalReaction, 2 * (input.span + input.rightOverhang) + 5);
  near(topOverhang.length, input.rightOverhang);
  near(
    Math.abs(topOverhang.start.moment),
    (2 * input.rightOverhang ** 2) / 2 + 5 * input.rightOverhang,
  );
  near(topOverhang.end.moment, 0);
  near(result.equilibrium.sumFy, 0);
  near(result.equilibrium.sumMoment, 0);
});

test("fixed, pinned, and mixed base supports stay stable in both frame examples", () => {
  for (const leftSupport of ["fixed", "pinned"] as const) {
    for (const rightSupport of ["fixed", "pinned"] as const) {
      const threeStory = makeDefaultThreeStoryInput();
      threeStory.leftSupport = leftSupport;
      threeStory.rightSupport = rightSupport;
      const threeStoryResult = solveFrame(threeStory);
      assert.ok(threeStoryResult.equilibrium.normalizedForceResidual < 1e-9);

      const inclined = makeDefaultInclinedFrameInput();
      inclined.supportA = leftSupport;
      inclined.supportD = rightSupport;
      const inclinedResult = solveFrame(inclined);
      assert.ok(inclinedResult.equilibrium.normalizedForceResidual < 1e-9);
      assert.ok(inclinedResult.equilibrium.normalizedMomentResidual < 1e-9);
    }
  }
});

test("inclined-frame point load changes support reactions by its full vertical magnitude", () => {
  const withoutPointLoad = makeDefaultInclinedFrameInput();
  withoutPointLoad.pointLoadAtB = 0;
  const withPointLoad = { ...withoutPointLoad, pointLoadAtB: 3 };
  const unloaded = solveFrame(withoutPointLoad);
  const loaded = solveFrame(withPointLoad);
  const verticalWithout = unloaded.reactions.reduce(
    (sum, reaction) => sum + reaction.fy,
    0,
  );
  const verticalWith = loaded.reactions.reduce(
    (sum, reaction) => sum + reaction.fy,
    0,
  );

  near(verticalWithout, 2.5 * 3.5);
  near(verticalWith - verticalWithout, 3);
  assert.ok(
    Math.abs(
      loaded.memberResults.find((member) => member.id === "AB")!.start.moment -
        unloaded.memberResults.find((member) => member.id === "AB")!.start.moment,
    ) > 0.01,
    "The point load at B should affect member end actions.",
  );
  assert.ok(loaded.equilibrium.normalizedForceResidual < 1e-10);
  assert.ok(loaded.equilibrium.normalizedMomentResidual < 1e-10);
});

test("invalid inclined geometry is rejected with a Spanish explanation", () => {
  const input = makeDefaultInclinedFrameInput();
  input.angleDegrees = 90;
  assert.throws(() => solveFrame(input), /ángulo CD debe estar entre 10° y 80°/i);
});

test("a negative downward point-load magnitude is rejected", () => {
  const input = makeDefaultInclinedFrameInput();
  input.pointLoadAtB = -1;
  assert.throws(() => solveFrame(input), /carga puntual descendente en B/i);
});
