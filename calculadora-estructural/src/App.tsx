import { useMemo, useState } from 'react';
import { AlertCircle, ArrowDown, ArrowRight, ChevronDown, CircleHelp, Compass, RotateCcw, Ruler, Scale, Sigma, SlidersHorizontal, Waves } from 'lucide-react';
import { solveFrame, type FrameInput as SolverFrameInput, type FrameResult } from './analysis/frame-solver';

type ModelKind = 'sway' | 'inclined';
type Support = 'fixed' | 'pinned';
type FrameInput = {
  model: ModelKind;
  E: number;
  A: number;
  sway: { span: number; rightOverhang: number; heights: [number, number, number]; columnI: [number, number, number]; beamI: number; leftBase: Support; rightBase: Support; floorLoads: [number, number, number]; tipLoads: [number, number, number]; lateralForces: [number, number, number] };
  inclined: { AB: number; BC: number; CDProjection: number; angle: number; IAB: number; IBC: number; ICD: number; supportA: Support; supportD: Support; beamLoad: number; pointLoadB: number };
};

const initialInput: FrameInput = {
  model: 'sway', E: 2_100_000, A: 0.03,
  sway: { span: 5, rightOverhang: 1.5, heights: [4, 3, 3], columnI: [0.004, 0.003, 0.002], beamI: 0.001, leftBase: 'fixed', rightBase: 'fixed', floorLoads: [3, 3, 2], tipLoads: [2, 2, 2], lateralForces: [8, 8, 6] },
  inclined: { AB: 3, BC: 3.5, CDProjection: 2, angle: 40, IAB: 0.001, IBC: 0.001, ICD: 0.001, supportA: 'fixed', supportD: 'fixed', beamLoad: 2.5, pointLoadB: 3 },
};

type AnyRecord = Record<string, unknown>;
const record = (v: unknown): AnyRecord => v && typeof v === 'object' ? v as AnyRecord : {};
const arr = (v: unknown): unknown[] => Array.isArray(v) ? v : [];
const val = (v: unknown, fallback = '—') => typeof v === 'number' ? Number(v.toFixed(3)).toString() : typeof v === 'string' ? v : fallback;

function Field({ label, value, unit, onChange, step = 'any', min = 0 }: { label: string; value: number; unit: string; onChange: (n: number) => void; step?: string; min?: number }) {
  return <label className="block">
    <span className="field-label">{label}</span>
    <span className="relative block">
      <input className="field-input mono pr-14" type="number" min={min} step={step} value={value} onChange={e => onChange(Number(e.target.value))} data-testid={`input-${label.toLowerCase().replaceAll(' ', '-')}`} />
      <span className="mono pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-[10px] text-[#8a918b]">{unit}</span>
    </span>
  </label>;
}

function SupportSelect({ label, value, onChange }: { label: string; value: Support; onChange: (s: Support) => void }) {
  return <label className="block">
    <span className="field-label">{label}</span>
    <span className="relative block">
      <select value={value} onChange={e => onChange(e.target.value as Support)} className="field-input appearance-none pr-8" data-testid={`select-${label.toLowerCase().replaceAll(' ', '-')}`}>
        <option value="fixed">Empotrado</option><option value="pinned">Articulado</option>
      </select><ChevronDown size={14} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-[#70807c]" />
    </span>
  </label>;
}

function SupportGlyph({ x, y, type }: { x: number; y: number; type: Support }) {
  return type === 'fixed'
    ? <g stroke="#83918a" strokeWidth="1.2" fill="none" aria-label="Apoyo empotrado">
        <line x1={x - 8} y1={y - 8} x2={x - 8} y2={y + 10}/>
        {[0,1,2,3].map(i=><line key={i} x1={x-8} y1={y-6+i*5} x2={x-15} y2={y-10+i*5}/>)}
      </g>
    : <g stroke="#83918a" strokeWidth="1.2" fill="none" aria-label="Apoyo articulado">
        <path d={`M${x-11} ${y+9} L${x} ${y+1} L${x+11} ${y+9} Z M${x-14} ${y+12} H${x+14}`}/>
      </g>;
}

function SvgFrame({ model, input }: { model: ModelKind; input: FrameInput }) {
  const sway = model === 'sway';
  const xLeft = 90, yBase = 292;
  const swayScaleX = Math.min(60, 300 / Math.max(input.sway.span + input.sway.rightOverhang, 0.1));
  const xRight = xLeft + input.sway.span * swayScaleX;
  const xOverhangEnd = xRight + input.sway.rightOverhang * swayScaleX;
  const heights = input.sway.heights;
  const stepY = [0, heights[0], heights[0] + heights[1], heights[0] + heights[1] + heights[2]];
  const scale = 205 / Math.max(stepY[3], sway ? 1 : input.inclined.AB + input.inclined.CDProjection);
  const floorY = stepY.map(h => yBase - h * scale);
  const maxY = floorY[3];
  const cdHorizontal = input.inclined.CDProjection * Math.tan(input.inclined.angle * Math.PI / 180);
  const inclinedScale = Math.min(185 / input.inclined.AB, 300 / (input.inclined.BC + cdHorizontal));
  const inclineA = { x: 110, y: 285 };
  const inclineB = { x: 110, y: inclineA.y - input.inclined.AB * inclinedScale };
  const inclineC = { x: inclineB.x + input.inclined.BC * inclinedScale, y: inclineB.y };
  const inclineD = {
    x: inclineC.x + cdHorizontal * inclinedScale,
    y: inclineA.y - (input.inclined.AB - input.inclined.CDProjection) * inclinedScale,
  };
  const dimText = <g className="mono" fill="#7e8983" fontSize="9" textAnchor="middle">
    {sway ? <><text x="260" y="319">{input.sway.span.toFixed(2)} m entre ejes · {input.sway.rightOverhang.toFixed(2)} m voladizo</text>{[0,1,2].map((i)=><text key={i} x="55" y={(floorY[i]+floorY[i+1])/2+3}>{heights[i].toFixed(2)} m</text>)}</> :
      <><text x={(inclineB.x+inclineC.x)/2} y="319">{input.inclined.BC.toFixed(2)} m · BC</text><text x="75" y={(inclineA.y+inclineB.y)/2}>{input.inclined.AB.toFixed(2)} m · AB</text><text x={Math.min(inclineD.x+38,490)} y={(inclineC.y+inclineD.y)/2}>{input.inclined.CDProjection.toFixed(2)} m proy.</text></>}
  </g>;
  return <div className="quiet-grid relative overflow-hidden rounded-lg border border-[#e1e2d9] bg-[#f9f8f2]">
    <div className="absolute left-3 top-3 section-kicker">Esquema · no a escala</div>
    <svg viewBox="0 0 520 340" className="block h-[330px] w-full" role="img" aria-label={sway ? 'Esquema de pórtico de tres niveles' : 'Esquema de pórtico inclinado'}>
      <defs><marker id="loadArrow" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="none" stroke="#ca7456" strokeWidth="1.5" /></marker></defs>
      {sway ? <>
        {[0,1,2,3].map(i=><g key={i}>
          {i > 0 && <><line x1={xLeft} y1={floorY[i]} x2={xRight} y2={floorY[i]} stroke="#2d6a61" strokeWidth="5" strokeLinecap="round"/><line x1={xRight} y1={floorY[i]} x2={xOverhangEnd} y2={floorY[i]} stroke="#2d6a61" strokeWidth="5" strokeLinecap="round"/></>}
          {i < 3 && <><line x1={xLeft} y1={floorY[i]} x2={xLeft} y2={floorY[i+1]} stroke="#35545a" strokeWidth="5" strokeLinecap="round"/><line x1={xRight} y1={floorY[i]} x2={xRight} y2={floorY[i+1]} stroke="#35545a" strokeWidth="5" strokeLinecap="round"/></>}
          <circle cx={xLeft} cy={floorY[i]} r="4" fill="#f9f8f2" stroke="#35545a" strokeWidth="2"/><circle cx={xRight} cy={floorY[i]} r="4" fill="#f9f8f2" stroke="#35545a" strokeWidth="2"/>
          {i > 0 && <circle cx={xOverhangEnd} cy={floorY[i]} r="4" fill="#f9f8f2" stroke="#35545a" strokeWidth="2"/>}
          {i > 0 && <g stroke="#ca7456" strokeWidth="1.4" markerEnd="url(#loadArrow)">{[0,1,2,3,4,5,6,7].map(j=>{const x=xLeft+12+j*(xOverhangEnd-xLeft-24)/7;return <line key={j} x1={x} y1={floorY[i]-25} x2={x} y2={floorY[i]-6}/>})}</g>}
          {i > 0 && <g stroke="#ca7456" strokeWidth="1.6" markerEnd="url(#loadArrow)"><line x1={xOverhangEnd} y1={floorY[i]-27} x2={xOverhangEnd} y2={floorY[i]-8}/></g>}
          {i > 0 && <g stroke="#ca7456" strokeWidth="1.5" markerEnd="url(#loadArrow)"><line x1={61} y1={floorY[i]+15} x2={89} y2={floorY[i]+15}/><line x1={430} y1={floorY[i]+15} x2={458} y2={floorY[i]+15}/></g>}
          {i > 0 && <text x={xOverhangEnd+7} y={floorY[i]-17} fill="#ca7456" fontSize="9" className="mono">P{i} ↓</text>}
        </g>)}
        <text x="462" y={maxY+7} fill="#ca7456" fontSize="9" className="mono">F/2 →</text>
        <g fill="#556b69" fontSize="9" className="mono"><text x={xLeft-15} y={floorY[3]-10}>D</text><text x={xRight+7} y={floorY[3]-10}>E</text><text x={xLeft-15} y={floorY[2]-10}>C</text><text x={xRight+7} y={floorY[2]-10}>F</text><text x={xLeft-15} y={floorY[1]-10}>B</text><text x={xRight+7} y={floorY[1]-10}>G</text><text x={xLeft-15} y={floorY[0]+16}>A</text><text x={xRight+7} y={floorY[0]+16}>H</text></g>
      </> : <>
        {(() => { const A=inclineA, B=inclineB, C=inclineC, D=inclineD; return <g>
          <line x1={A.x} y1={A.y} x2={B.x} y2={B.y} stroke="#35545a" strokeWidth="5" strokeLinecap="round"/><line x1={B.x} y1={B.y} x2={C.x} y2={C.y} stroke="#2d6a61" strokeWidth="5" strokeLinecap="round"/><line x1={C.x} y1={C.y} x2={D.x} y2={D.y} stroke="#35545a" strokeWidth="5" strokeLinecap="round"/>
          {[A,B,C,D].map((p,i)=><circle key={i} cx={p.x} cy={p.y} r="4" fill="#f9f8f2" stroke="#35545a" strokeWidth="2"/> )}
          <g stroke="#ca7456" strokeWidth="1.4" markerEnd="url(#loadArrow)">{[0,1,2,3,4,5].map(j=>{const x=B.x+(C.x-B.x)*(j+1)/7;return <line key={j} x1={x} y1={B.y-27} x2={x} y2={B.y-7}/>})}</g>
          <line x1={B.x} y1={B.y-28} x2={B.x} y2={B.y-7} stroke="#ca7456" strokeWidth="1.6" markerEnd="url(#loadArrow)"/>
          <g fill="#556b69" fontSize="10" className="mono"><text x={A.x-15} y={A.y+16}>A</text><text x={B.x-15} y={B.y-10}>B</text><text x={C.x+8} y={C.y-8}>C</text><text x={D.x+8} y={D.y+14}>D</text></g>
          <text x={B.x+9} y={B.y-31} fill="#ca7456" fontSize="9" className="mono">P ↓</text>
        </g> })()}
      </>}
      {dimText}
      {sway
        ? <><SupportGlyph x={xLeft} y={yBase} type={input.sway.leftBase}/><SupportGlyph x={xRight} y={yBase} type={input.sway.rightBase}/></>
        : <><SupportGlyph x={inclineA.x} y={inclineA.y} type={input.inclined.supportA}/><SupportGlyph x={inclineD.x} y={inclineD.y} type={input.inclined.supportD}/></>}
      <text x="257" y="24" textAnchor="middle" fill="#7d8983" fontSize="9" className="mono">{sway ? '3 niveles · desplazamiento lateral · voladizos derechos' : `CD · ${input.inclined.angle}° respecto a vertical`}</text>
    </svg>
  </div>;
}

function Results({ result, model }: { result: FrameResult; model: ModelKind }) {
  const r = record(result);
  const members = arr(r.memberResults);
  const reactions = arr(r.reactions);
  const actionsFor = (m: unknown) => {
    const item = record(m);
    const i = record(item.start);
    const j = record(item.end);
    return { id: val(item.id, 'Miembro'), length: val(item.length), i, j, diagram: arr(item.diagram) };
  };
  const resultMembers = members.map(actionsFor);
  const formatActions = (action: AnyRecord) => `N ${val(action.axial)} · V ${val(action.shear)} · M ${val(action.moment)}`;
  const shearData = resultMembers.map(m => ({ id: m.id, points: m.diagram }));
  const momentData = resultMembers.map(m => ({ id: m.id, points: m.diagram }));
  return <section className="mt-5 space-y-4 appear" data-testid="results-panel">
    <div className="flex flex-wrap items-end justify-between gap-3 border-b border-[#dfddd2] pb-3">
      <div><div className="section-kicker">Salida del análisis</div><h2 className="mt-1 text-[22px] font-semibold tracking-[-.04em] text-[#20343b]">Resultados del modelo</h2></div>
      <div className="mono flex items-center gap-2 text-[10px] text-[#55736c]"><span className="h-2 w-2 rounded-full bg-[#4c8b70]"/> Cálculo local completado</div>
    </div>
    {typeof r.summary === 'string' && <p className="text-sm leading-6 text-[#526267]">{r.summary}</p>}
    <div className="grid gap-4 lg:grid-cols-[1.3fr_.7fr]">
      <div className="panel overflow-hidden">
        <div className="border-b border-[#e5e3da] px-4 py-3"><div className="section-kicker">Acciones de extremo</div></div>
        {resultMembers.length ? <div className="overflow-x-auto"><table className="w-full text-left text-xs"><thead className="bg-[#f4f2eb] text-[10px] uppercase tracking-[.08em] text-[#71807c]"><tr><th className="px-4 py-2.5">Barra</th><th className="px-4 py-2.5">Longitud</th><th className="px-4 py-2.5">Extremo inicial · N / V / M</th><th className="px-4 py-2.5">Extremo final · N / V / M</th></tr></thead><tbody>{resultMembers.map((m,i)=><tr key={`${m.id}-${i}`} className="border-t border-[#eceae2]"><td className="px-4 py-3 font-semibold">{m.id}</td><td className="mono px-4 py-3">{m.length} m</td><td className="mono px-4 py-3 text-[10px] leading-5">{formatActions(m.i)}</td><td className="mono px-4 py-3 text-[10px] leading-5">{formatActions(m.j)}</td></tr>)}</tbody></table></div> :
        <p className="px-4 py-4 text-sm text-[#75817e]">El motor no devolvió acciones de extremo identificables.</p>}
      </div>
      <div className="panel p-4">
        <div className="section-kicker">Reacciones de apoyo</div>
        {reactions.length ? <div className="mt-3 divide-y divide-[#e6e4dc]">{reactions.map((reaction,i)=>{const q=record(reaction);return <div key={i} className="py-2 text-xs"><div className="font-semibold">{val(q.nodeId, `Apoyo ${i+1}`)}</div><div className="mono mt-1 text-[10px] leading-5 text-[#54666a]">Fx {val(q.fx)} ton · Fy {val(q.fy)} ton · M {val(q.moment)} ton·m</div></div>})}</div> :
        <p className="mt-3 text-sm text-[#75817e]">El motor no devolvió reacciones identificables.</p>}
      </div>
    </div>
    <div className="grid gap-4 lg:grid-cols-2">
      {(['cortante','momento'] as const).map((label,idx)=>{
        const chartData = idx===0 ? shearData : momentData;
        const axis = idx===0 ? 'shear' : 'moment';
        const unit = idx===0 ? 'ton' : 'ton·m';
        const sharedScale = Math.max(
          ...chartData.flatMap(item => item.points.map(point => Math.abs(Number(record(point)[axis])))),
          1e-9,
        );
        return <div className="panel p-4" key={label}>
          <div className="flex items-center justify-between">
            <div><div className="section-kicker">Diagrama {label}</div><div className="mt-1 text-sm font-semibold">{label==='cortante'?'V · fuerza cortante':'M · momento flector'}</div></div>
            <div className="text-right"><div className="mono text-[10px] text-[#788681]">{unit}</div><div className="mono text-[9px] text-[#788681]">escala común ±{val(sharedScale)}</div></div>
          </div>
          {chartData.length ? <div className="mt-4 space-y-2">{chartData.map((d,i)=>{
            const q=record(d);
            const points=arr(q.points).map(record);
            const values=points.map(p=>Number(p[axis]));
            const maximum=Math.max(...values);
            const minimum=Math.min(...values);
            const length=Number(resultMembers[i]?.length || 1);
            const poly=points.map(p=>`${30+(Number(p.x)/length)*260},${41-(Number(p[axis])/sharedScale)*29}`).join(' ');
            return <div key={i} data-testid={`diagram-${label}-${String(q.id).toLowerCase()}`} className="rounded bg-[#f6f5ef] p-2">
              <div className="mono mb-1 flex flex-wrap justify-between gap-x-3 text-[10px] text-[#71807c]">
                <span>{val(q.id, `Elemento ${i+1}`)} · 0–{val(length)} m</span>
                <span>máx. {val(maximum)} / mín. {val(minimum)} {unit}</span>
              </div>
              <svg viewBox="0 0 320 82" className="h-[70px] w-full" role="img" aria-label={`Diagrama de ${label} para ${val(q.id)}`}>
                <line x1="30" y1="41" x2="290" y2="41" stroke="#c4c9c1"/>
                {points.length>1?<polyline points={poly} fill="none" stroke={idx===0?'#ca7456':'#2d6a61'} strokeWidth="2.2" strokeLinejoin="round"/>:<text x="160" y="46" textAnchor="middle" fill="#7d8983" fontSize="10">Puntos de diagrama no disponibles</text>}
                <text x="30" y="76" fill="#8b9690" fontSize="8">0</text><text x="290" y="76" textAnchor="end" fill="#8b9690" fontSize="8">{val(length)} m</text>
              </svg>
            </div>;
          })}</div> :
          <div className="mt-4 flex h-24 items-center justify-center rounded bg-[#f6f5ef] px-4 text-center text-xs leading-5 text-[#788681]">El motor no ha proporcionado puntos para este diagrama.</div>}
        </div>;
      })}
    </div>
    <div className="panel p-4">
      <div className="section-kicker">Supuestos del solver</div>
      <p className="mt-2 text-xs leading-5 text-[#69797a]">Modelo elástico lineal de primer orden con barras prismáticas de pórtico 2D Euler–Bernoulli, nudos rígidos y sin deformación por cortante ni efectos de segundo orden. El equilibrio de nudos y el desplazamiento lateral se resuelven simultáneamente por rigidez directa. La tabla muestra acciones de barra sobre los nudos (momento antihorario positivo); los diagramas muestran fuerzas internas, con N positivo a tracción y M positivo sagante. Fuerzas y reacciones están en ejes globales; los diagramas, en ejes locales.</p>
      <div className="mt-3 grid gap-2 text-[10px] text-[#69797a] sm:grid-cols-3"><span>Residuo de fuerza: <b className="mono font-medium">{val(record(r.equilibrium).normalizedForceResidual)}</b></span><span>Residuo de momento: <b className="mono font-medium">{val(record(r.equilibrium).normalizedMomentResidual)}</b></span><span>Equilibrio resultante: <b className="mono font-medium">{val(record(r.equilibrium).sumFx)} Fx · {val(record(r.equilibrium).sumFy)} Fy</b></span></div>
    </div>
    {arr(r.storyDrifts).length>0 && <div className="panel overflow-hidden"><div className="border-b border-[#e5e3da] px-4 py-3"><div className="section-kicker">Deriva por nivel</div></div><div className="grid gap-px bg-[#e9e7df] sm:grid-cols-3">{arr(r.storyDrifts).map((drift,i)=>{const q=record(drift);return <div key={i} className="bg-[#fcfbf7] px-4 py-3"><div className="text-xs font-semibold">{val(q.id,`Nivel ${i+1}`)}</div><div className="mono mt-1 text-[10px] text-[#647572]">{val(q.horizontalDisplacement)} m · razón {val(q.driftRatio)}</div></div>})}</div></div>}
    <div className="text-[10px] leading-5 text-[#78827f]">Modelo: {model==='sway'?'pórtico de tres niveles':'pórtico inclinado'} · Conectividad definida por el ejemplo seleccionado. Los signos se interpretan según las convenciones del solver.</div>
  </section>;
}

function App() {
  const [input, setInput] = useState<FrameInput>(initialInput);
  const [result, setResult] = useState<FrameResult>();
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const update = <K extends keyof FrameInput>(key: K, value: FrameInput[K]) => setInput(prev=>({...prev,[key]:value}));
  const currentModel = input.model;
  const included = useMemo(() => currentModel === 'sway' ? [
    'Nudos de base A/H; columnas y vigas en los tres niveles con voladizo a la derecha.',
    'La carga uniforme cubre cada luz entre ejes y su voladizo.',
    'Carga puntual descendente en la punta de cada voladizo y fuerzas laterales por nivel.',
  ] : [
    'Conectividad A–B–C–D; A y D son apoyos, B y C son nudos rígidos.',
    'CD se define por su proyección vertical y ángulo respecto a la vertical.',
    'Carga uniforme descendente en BC y carga puntual descendente en B.',
  ], [currentModel]);

  const setNested = (section: 'sway'|'inclined', key: string, value: unknown) => setInput(prev=>({...prev,[section]:{...prev[section], [key]:value}} as FrameInput));
  const solve = async () => {
    setBusy(true); setError(''); setResult(undefined);
    try {
      const solverInput: SolverFrameInput = input.model === 'sway' ? {
        type: 'three-story', E: input.E, A: input.A, span: input.sway.span,
        rightOverhang: input.sway.rightOverhang, tipLoads: input.sway.tipLoads,
        storyHeights: input.sway.heights, columnInertias: input.sway.columnI,
        beamInertia: input.sway.beamI, leftSupport: input.sway.leftBase,
        rightSupport: input.sway.rightBase, floorLoads: input.sway.floorLoads,
        lateralLoads: input.sway.lateralForces,
      } : {
        type: 'inclined', E: input.E, A: input.A, ABheight: input.inclined.AB,
        BCspan: input.inclined.BC, CDverticalProjection: input.inclined.CDProjection,
        angleDegrees: input.inclined.angle, inertias: { AB: input.inclined.IAB, BC: input.inclined.IBC, CD: input.inclined.ICD },
        supportA: input.inclined.supportA, supportD: input.inclined.supportD,
        distributedLoad: input.inclined.beamLoad, pointLoadAtB: input.inclined.pointLoadB,
      };
      const solved = solveFrame(solverInput);
      if (solved == null) throw new Error('El motor no devolvió resultados para este modelo.');
      setResult(solved);
    } catch (e) {
      const message = e instanceof Error ? e.message : 'No se pudo resolver el modelo.';
      setError(message);
    } finally { setBusy(false); }
  };
  const reset = () => { setInput(initialInput); setResult(undefined); setError(''); };

  return <main className="app-shell text-[#20343b]">
    <header className="topbar bg-[#fbfaf5]">
      <div className="mx-auto flex max-w-[1500px] items-center justify-between gap-4 px-5 py-4 md:px-8">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-[#2d6a61] text-[#f8f5ed]"><Compass size={19} strokeWidth={1.7}/></div>
          <div><div className="font-[Manrope] text-[15px] font-extrabold tracking-[-.04em]">Calculadora estructural</div><div className="mono mt-0.5 text-[9px] uppercase tracking-[.13em] text-[#86918c]">cuaderno de aprendizaje</div></div>
        </div>
        <div className="hidden items-center gap-2 rounded-full border border-[#dfddd2] bg-[#f6f5ef] px-3 py-1.5 text-[10px] text-[#677570] sm:flex"><span className="h-1.5 w-1.5 rounded-full bg-[#5a8b72]"/> Solo cálculo local <span className="mx-1 text-[#c1c4bc]">/</span> Sin guardar</div>
      </div>
    </header>
    <div className="mx-auto max-w-[1500px] px-4 pb-12 pt-7 md:px-8 md:pt-10">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div><div className="section-kicker">Banco de pruebas · 01</div><h1 className="mt-2 font-[Manrope] text-[clamp(28px,4vw,42px)] font-semibold leading-[1.05] tracking-[-.055em]">Sigue el camino de las cargas.</h1><p className="mt-2 max-w-2xl text-sm leading-6 text-[#68777a]">Edita un ejemplo de pórtico, resuélvelo en tu navegador y observa cómo las condiciones y las cargas cambian la respuesta.</p></div>
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[.12em] text-[#74817d]"><span className="mono text-[#ba7156]">LAB / 01</span><span className="h-px w-8 bg-[#d9d8ce]"/><span>Modelos planos</span></div>
      </div>
      <div className="grid gap-5 xl:grid-cols-[minmax(390px,.87fr)_minmax(440px,1.13fr)]">
        <section className="space-y-4">
          <div className="panel overflow-hidden">
            <div className="flex items-center justify-between border-b border-[#e4e2d9] px-4 py-3.5 md:px-5">
              <div><div className="section-kicker">01 · Elige un ejemplo</div><h2 className="mt-1 text-sm font-semibold">Geometría y sistema</h2></div><SlidersHorizontal size={16} className="text-[#798883]"/>
            </div>
            <div className="grid grid-cols-2 gap-2 p-3">
              <button onClick={()=>{update('model','sway');setResult(undefined);setError('')}} data-testid="button-model-sway" className={`rounded-md border px-3 py-3 text-left transition-colors ${currentModel==='sway'?'border-[#2d6a61] bg-[#eaf1eb]':'border-[#e4e2d9] bg-[#faf9f4] hover:border-[#aabbb3]'}`}>
                <span className="mono text-[9px] uppercase tracking-[.1em] text-[#7b8983]">Ejemplo A</span><span className="mt-1 block text-xs font-semibold">Pórtico con desplazamiento</span><span className="mt-1 block text-[10px] text-[#7a8782]">3 niveles · acciones laterales</span>
              </button>
              <button onClick={()=>{update('model','inclined');setResult(undefined);setError('')}} data-testid="button-model-inclined" className={`rounded-md border px-3 py-3 text-left transition-colors ${currentModel==='inclined'?'border-[#2d6a61] bg-[#eaf1eb]':'border-[#e4e2d9] bg-[#faf9f4] hover:border-[#aabbb3]'}`}>
                <span className="mono text-[9px] uppercase tracking-[.1em] text-[#7b8983]">Ejemplo B</span><span className="mt-1 block text-xs font-semibold">Pórtico con barra inclinada</span><span className="mt-1 block text-[10px] text-[#7a8782]">3 barras · geometría angular</span>
              </button>
            </div>
          </div>
          <div className="panel p-4 md:p-5">
            <div className="mb-4 flex items-start justify-between"><div><div className="section-kicker">02 · Parámetros compartidos</div><h2 className="mt-1 text-sm font-semibold">Propiedades del material y sección</h2></div><Sigma size={16} className="text-[#75847e]"/></div>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Módulo E" value={input.E} unit="ton/m²" onChange={n=>update('E',n)} />
              <Field label="Área A" value={input.A} unit="m²" onChange={n=>update('A',n)} />
            </div>
          </div>
          {currentModel==='sway' ? <div className="panel p-4 md:p-5">
            <div className="mb-4"><div className="section-kicker">03 · Geometría y rigidez</div><h2 className="mt-1 text-sm font-semibold">Pórtico de tres niveles</h2></div>
            <div className="grid grid-cols-2 gap-3"><Field label="Luz entre ejes" value={input.sway.span} unit="m" onChange={n=>setNested('sway','span',n)}/><Field label="Voladizo derecho" value={input.sway.rightOverhang} unit="m" onChange={n=>setNested('sway','rightOverhang',n)}/><Field label="Inercia de vigas" value={input.sway.beamI} unit="m⁴" onChange={n=>setNested('sway','beamI',n)}/></div>
            <div className="mt-4 grid grid-cols-3 gap-2">{input.sway.heights.map((v,i)=><Field key={i} label={`Altura nivel ${i+1}`} value={v} unit="m" onChange={n=>{const a=[...input.sway.heights] as [number,number,number];a[i]=n;setNested('sway','heights',a)}} />)}</div>
            <div className="mt-4 grid grid-cols-3 gap-2">{input.sway.columnI.map((v,i)=><Field key={i} label={`I columnas N${i+1}`} value={v} unit="m⁴" onChange={n=>{const a=[...input.sway.columnI] as [number,number,number];a[i]=n;setNested('sway','columnI',a)}} />)}</div>
            <div className="mt-4 grid grid-cols-2 gap-3"><SupportSelect label="Apoyo izquierdo" value={input.sway.leftBase} onChange={v=>setNested('sway','leftBase',v)}/><SupportSelect label="Apoyo derecho" value={input.sway.rightBase} onChange={v=>setNested('sway','rightBase',v)}/></div>
            <div className="my-4 h-px bg-[#e8e6de]"/>
            <div className="mb-3 flex items-center justify-between"><div><div className="section-kicker">04 · Acciones aplicadas</div><div className="mt-1 text-xs text-[#74817d]">Carga descendente positiva · fuerza lateral a la derecha positiva</div></div><Scale size={15} className="text-[#bd755b]"/></div>
            <div className="space-y-2">{input.sway.floorLoads.map((v,i)=><div key={i} className="grid grid-cols-[1fr_1fr] items-end gap-3"><div className="flex items-center gap-2 pb-2 text-xs text-[#61716e]"><ArrowDown size={13} className="text-[#ca7456]"/>Carga nivel {i+1}</div><Field label="w distribuida" value={v} unit="ton/m" onChange={n=>{const a=[...input.sway.floorLoads] as [number,number,number];a[i]=n;setNested('sway','floorLoads',a)}}/></div>)}</div>
            <div className="mt-3 space-y-2">{input.sway.tipLoads.map((v,i)=><div key={i} className="grid grid-cols-[1fr_1fr] items-end gap-3"><div className="flex items-center gap-2 pb-2 text-xs text-[#61716e]"><ArrowDown size={13} className="text-[#ca7456]"/>Punta voladizo nivel {i+1}</div><Field label={`P en punta N${i+1}`} value={v} unit="ton" onChange={n=>{const a=[...input.sway.tipLoads] as [number,number,number];a[i]=n;setNested('sway','tipLoads',a)}}/></div>)}</div>
            <div className="mt-3 space-y-2">{input.sway.lateralForces.map((v,i)=><div key={i} className="grid grid-cols-[1fr_1fr] items-end gap-3"><div className="flex items-center gap-2 pb-2 text-xs text-[#61716e]"><ArrowRight size={13} className="text-[#ca7456]"/>Fuerza de piso {i+1}</div><Field label="F lateral" value={v} unit="ton" onChange={n=>{const a=[...input.sway.lateralForces] as [number,number,number];a[i]=n;setNested('sway','lateralForces',a)}}/></div>)}</div>
          </div> : <div className="panel p-4 md:p-5">
            <div className="mb-4"><div className="section-kicker">03 · Geometría y rigidez</div><h2 className="mt-1 text-sm font-semibold">Pórtico inclinado</h2></div>
            <div className="grid grid-cols-2 gap-3"><Field label="AB · vertical" value={input.inclined.AB} unit="m" onChange={n=>setNested('inclined','AB',n)}/><Field label="BC · luz" value={input.inclined.BC} unit="m" onChange={n=>setNested('inclined','BC',n)}/><Field label="Proyección vertical CD" value={input.inclined.CDProjection} unit="m" onChange={n=>setNested('inclined','CDProjection',n)}/><Field label="Ángulo CD / vertical" value={input.inclined.angle} unit="°" min={1} onChange={n=>setNested('inclined','angle',n)}/></div>
            <div className="mt-3 flex items-center justify-between rounded-md bg-[#f3f2eb] px-3 py-2.5 text-xs"><span className="text-[#6e7b77]">Longitud real de CD = proyección / cos(θ)</span><span className="mono font-medium text-[#314a49]">{(input.inclined.CDProjection / Math.cos(input.inclined.angle * Math.PI / 180)).toFixed(3)} m</span></div>
            <div className="mt-4 grid grid-cols-3 gap-2"><Field label="I · AB" value={input.inclined.IAB} unit="m⁴" onChange={n=>setNested('inclined','IAB',n)}/><Field label="I · BC" value={input.inclined.IBC} unit="m⁴" onChange={n=>setNested('inclined','IBC',n)}/><Field label="I · CD" value={input.inclined.ICD} unit="m⁴" onChange={n=>setNested('inclined','ICD',n)}/></div>
            <div className="mt-4 grid grid-cols-2 gap-3"><SupportSelect label="Apoyo A" value={input.inclined.supportA} onChange={v=>setNested('inclined','supportA',v)}/><SupportSelect label="Apoyo D" value={input.inclined.supportD} onChange={v=>setNested('inclined','supportD',v)}/></div>
            <div className="my-4 h-px bg-[#e8e6de]"/>
            <div className="mb-3"><div className="section-kicker">04 · Acciones aplicadas</div><div className="mt-1 text-xs text-[#74817d]">Carga descendente positiva · P aplicada en el nudo B</div></div>
            <div className="grid grid-cols-2 gap-3"><Field label="Carga uniforme en BC" value={input.inclined.beamLoad} unit="ton/m" onChange={n=>setNested('inclined','beamLoad',n)}/><Field label="Carga puntual en B" value={input.inclined.pointLoadB} unit="ton" onChange={n=>setNested('inclined','pointLoadB',n)}/></div>
          </div>}
          <div className="flex flex-wrap gap-3">
            <button onClick={solve} disabled={busy} data-testid="button-solve" className="flex min-h-11 flex-1 items-center justify-center gap-2 rounded-md bg-[#2d6a61] px-5 text-sm font-semibold text-[#fffdf6] transition hover:bg-[#23584f] disabled:cursor-wait disabled:opacity-65">{busy?<><span className="h-4 w-4 animate-pulse rounded-full border-2 border-white/40 border-t-white"/>Resolviendo modelo…</>:<><Waves size={16}/>Resolver modelo local</>}</button>
            <button onClick={reset} data-testid="button-reset" className="flex min-h-11 items-center justify-center gap-2 rounded-md border border-[#d9d8ce] bg-[#fbfaf5] px-4 text-xs font-semibold text-[#65736f] hover:border-[#93aaa0]"><RotateCcw size={14}/>Restablecer</button>
          </div>
          {error && <div role="alert" data-testid="status-solver-error" className="flex gap-3 rounded-md border border-[#e5c8b8] bg-[#fcf0e9] p-3.5 text-xs leading-5 text-[#824c39]"><AlertCircle size={17} className="mt-0.5 shrink-0"/><div><div className="font-semibold">No se pudo resolver el modelo</div><div className="mt-0.5">{error}</div></div></div>}
        </section>
        <section className="min-w-0">
          <div className="panel overflow-hidden">
            <div className="flex items-center justify-between border-b border-[#e4e2d9] px-4 py-3.5 md:px-5"><div><div className="section-kicker">05 · Lectura del modelo</div><h2 className="mt-1 text-sm font-semibold">Conectividad y acciones</h2></div><Ruler size={16} className="text-[#798883]"/></div>
            <div className="p-3 md:p-4"><SvgFrame model={currentModel} input={input}/></div>
            <div className="grid gap-0 border-t border-[#e4e2d9] md:grid-cols-[1fr_1fr]">
              <div className="border-b border-[#e4e2d9] p-4 md:border-b-0 md:border-r md:px-5"><div className="section-kicker">Conectividad asumida</div><ul className="mt-2 space-y-1.5 text-[11px] leading-5 text-[#68777a]">{included.map((line,i)=><li className="flex gap-2" key={i}><span className="text-[#2d6a61]">•</span>{line}</li>)}</ul></div>
              <div className="p-4 md:px-5"><div className="section-kicker">Convención de signos</div><div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-2 text-[10px] text-[#68777a]"><span className="flex items-center gap-1.5"><ArrowDown size={12} className="text-[#ca7456]"/>Vertical descendente: +</span><span className="flex items-center gap-1.5"><ArrowRight size={12} className="text-[#ca7456]"/>Horizontal derecha: +</span><span>N: tracción positiva</span><span>V: eje local +y</span><span>M: momento sagante positivo</span></div></div>
            </div>
          </div>
          {result ? <Results result={result} model={currentModel}/> : <div className="mt-4 rounded-lg border border-dashed border-[#d7d8cd] bg-[#f4f2eb]/70 px-5 py-5">
            <div className="flex gap-3"><div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#e6ede5] text-[#2d6a61]"><CircleHelp size={16}/></div><div><div className="text-xs font-semibold">Una herramienta para aprender, no para construir.</div><p className="mt-1 max-w-xl text-[11px] leading-5 text-[#75817d]">Los resultados dependen de la idealización, la conectividad y las convenciones adoptadas. Verifica las hipótesis antes de interpretar las salidas. Este entorno no es software certificado ni sustituye el criterio de un profesional competente.</p></div></div>
          </div>}
        </section>
      </div>
      <footer className="mt-9 flex flex-wrap items-center justify-between gap-2 border-t border-[#dfddd2] pt-4 text-[10px] text-[#89918c]"><span>Calculadora estructural <span className="mx-1">·</span> laboratorio educativo de análisis de pórticos</span><span className="mono">UNIDADES DE FUERZA: ton <span className="mx-1">/</span> LONGITUD: m <span className="mx-1">/</span> MOMENTO: ton·m</span></footer>
    </div>
  </main>;
}

export default App;
