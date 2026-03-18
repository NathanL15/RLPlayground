import { Canvas } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";

/* ── Arena constants (must match backend) ────────────────────────── */
const L = 36;      // field length
const W = 24;      // field width
const H = 8;       // ceiling height
const GW = 8;      // goal width
const GH = 3.2;    // goal height
const GD = 2.5;    // goal depth

const BLUE = "#1d4ed8";
const ORANGE = "#ea580c";

/* Boost pad positions (must match backend order: big first, then small) */
const BIG_PADS = [
  [-14.4, -8.0], [-14.4, 8.0],
  [14.4, -8.0], [14.4, 8.0],
  [0.0, -10.8], [0.0, 10.8],
];
const SMALL_PADS = [
  [-3.6, 0.0], [3.6, 0.0],
  [0.0, -4.0], [0.0, 4.0],
  [-7.2, -4.8], [-7.2, 4.8],
  [7.2, -4.8], [7.2, 4.8],
  [-10.8, -4.8], [-10.8, 4.8],
  [10.8, -4.8], [10.8, 4.8],
  [-14.4, 0.0], [14.4, 0.0],
  [-7.2, 0.0], [7.2, 0.0],
  [-3.6, -7.2], [-3.6, 7.2],
  [3.6, -7.2], [3.6, 7.2],
];

/* ── Field surface + markings ────────────────────────────────────── */
function Field() {
  return (
    <group>
      {/* Main grass surface */}
      <mesh position={[0, 0, -0.05]} receiveShadow>
        <boxGeometry args={[L + GD * 2, W, 0.1]} />
        <meshStandardMaterial color="#1a5c2e" />
      </mesh>
      {/* Darker border */}
      <mesh position={[0, 0, -0.07]} receiveShadow>
        <boxGeometry args={[L + GD * 2 + 2, W + 2, 0.08]} />
        <meshStandardMaterial color="#0f3a1c" />
      </mesh>

      {/* Center line */}
      <mesh position={[0, 0, 0.01]}>
        <boxGeometry args={[0.1, W - 0.5, 0.01]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.3} />
      </mesh>
      {/* Center circle */}
      <mesh position={[0, 0, 0.01]}>
        <ringGeometry args={[5.6, 5.8, 64]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.25} side={THREE.DoubleSide} />
      </mesh>
      {/* Center dot */}
      <mesh position={[0, 0, 0.012]}>
        <circleGeometry args={[0.3, 32]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.35} side={THREE.DoubleSide} />
      </mesh>

      {/* Goal area markings */}
      <GoalAreaLines x={-L / 2} />
      <GoalAreaLines x={L / 2} />

      {/* Subtle half-field team tint */}
      <mesh position={[-L / 4, 0, 0.003]}>
        <boxGeometry args={[L / 2, W - 0.2, 0.001]} />
        <meshBasicMaterial color={BLUE} transparent opacity={0.035} />
      </mesh>
      <mesh position={[L / 4, 0, 0.003]}>
        <boxGeometry args={[L / 2, W - 0.2, 0.001]} />
        <meshBasicMaterial color={ORANGE} transparent opacity={0.035} />
      </mesh>
    </group>
  );
}

function GoalAreaLines({ x }) {
  const sign = Math.sign(x);
  const depth = 4.5;
  const w = GW + 2;
  const t = 0.08;
  const z = 0.011;
  return (
    <group>
      <mesh position={[x - sign * depth, 0, z]}>
        <boxGeometry args={[t, w, 0.01]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.2} />
      </mesh>
      <mesh position={[x - sign * (depth / 2), w / 2, z]}>
        <boxGeometry args={[depth, t, 0.01]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.2} />
      </mesh>
      <mesh position={[x - sign * (depth / 2), -w / 2, z]}>
        <boxGeometry args={[depth, t, 0.01]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.2} />
      </mesh>
    </group>
  );
}

/* ── Walls ────────────────────────────────────────────────────────── */
function Walls() {
  const opacity = 0.5;
  const thick = 0.2;
  const sideW = (W - GW) / 2;          // width of each back-wall segment beside the goal
  const topH = H - GH;                 // back-wall height above goal

  return (
    <group>
      {/* Side walls */}
      <mesh position={[0, W / 2, H / 2]}>
        <boxGeometry args={[L, thick, H]} />
        <meshStandardMaterial color="#334155" transparent opacity={opacity} />
      </mesh>
      <mesh position={[0, -W / 2, H / 2]}>
        <boxGeometry args={[L, thick, H]} />
        <meshStandardMaterial color="#334155" transparent opacity={opacity} />
      </mesh>

      {/* Blue back wall (X = -L/2) */}
      <mesh position={[-L / 2, -(GW / 2 + sideW / 2), H / 2]}>
        <boxGeometry args={[thick, sideW, H]} />
        <meshStandardMaterial color={BLUE} transparent opacity={opacity * 0.75} />
      </mesh>
      <mesh position={[-L / 2, GW / 2 + sideW / 2, H / 2]}>
        <boxGeometry args={[thick, sideW, H]} />
        <meshStandardMaterial color={BLUE} transparent opacity={opacity * 0.75} />
      </mesh>
      <mesh position={[-L / 2, 0, GH + topH / 2]}>
        <boxGeometry args={[thick, GW, topH]} />
        <meshStandardMaterial color={BLUE} transparent opacity={opacity * 0.55} />
      </mesh>

      {/* Orange back wall (X = L/2) */}
      <mesh position={[L / 2, -(GW / 2 + sideW / 2), H / 2]}>
        <boxGeometry args={[thick, sideW, H]} />
        <meshStandardMaterial color={ORANGE} transparent opacity={opacity * 0.75} />
      </mesh>
      <mesh position={[L / 2, GW / 2 + sideW / 2, H / 2]}>
        <boxGeometry args={[thick, sideW, H]} />
        <meshStandardMaterial color={ORANGE} transparent opacity={opacity * 0.75} />
      </mesh>
      <mesh position={[L / 2, 0, GH + topH / 2]}>
        <boxGeometry args={[thick, GW, topH]} />
        <meshStandardMaterial color={ORANGE} transparent opacity={opacity * 0.55} />
      </mesh>

    </group>
  );
}

/* ── Goals ─────────────────────────────────────────────────────────── */
function Goal({ side }) {
  const x = side === "blue" ? -L / 2 : L / 2;
  const ds = side === "blue" ? -1 : 1;
  const color = side === "blue" ? BLUE : ORANGE;
  const r = 0.14;

  return (
    <group>
      {/* Posts (vertical) */}
      {[-GW / 2, GW / 2].map((py, i) => (
        <mesh key={i} position={[x, py, GH / 2]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[r, r, GH, 10]} />
          <meshStandardMaterial color="#cccccc" metalness={0.85} roughness={0.15} />
        </mesh>
      ))}

      {/* Crossbar (horizontal along Y) */}
      <mesh position={[x, 0, GH]}>
        <cylinderGeometry args={[r, r, GW, 10]} />
        <meshStandardMaterial color="#cccccc" metalness={0.85} roughness={0.15} />
      </mesh>

      {/* Net back */}
      <mesh position={[x + ds * GD, 0, GH / 2]}>
        <boxGeometry args={[0.05, GW, GH]} />
        <meshStandardMaterial color={color} transparent opacity={0.12} side={THREE.DoubleSide} />
      </mesh>
      {/* Net sides */}
      {[-GW / 2, GW / 2].map((py, i) => (
        <mesh key={`ns${i}`} position={[x + ds * GD / 2, py, GH / 2]}>
          <boxGeometry args={[GD, 0.05, GH]} />
          <meshStandardMaterial color={color} transparent opacity={0.09} side={THREE.DoubleSide} />
        </mesh>
      ))}
      {/* Net top */}
      <mesh position={[x + ds * GD / 2, 0, GH]}>
        <boxGeometry args={[GD, GW, 0.05]} />
        <meshStandardMaterial color={color} transparent opacity={0.08} side={THREE.DoubleSide} />
      </mesh>
      {/* Goal floor tint */}
      <mesh position={[x + ds * GD / 2, 0, 0.004]}>
        <boxGeometry args={[GD, GW, 0.01]} />
        <meshStandardMaterial color={color} transparent opacity={0.12} />
      </mesh>
      {/* Goal line */}
      <mesh position={[x, 0, 0.015]}>
        <boxGeometry args={[0.12, GW, 0.01]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.0} />
      </mesh>
      {/* Goal area light */}
      <pointLight position={[x + ds * 1.5, 0, 3]} intensity={0.4} color={color} distance={10} />
    </group>
  );
}

/* ── Boost pads ───────────────────────────────────────────────────── */
function BoostPadMesh({ position, isBig, active }) {
  const radius = isBig ? 1.1 : 0.55;
  const height = isBig ? 0.14 : 0.07;
  const segments = isBig ? 6 : 16;
  const glowColor = "#f59e0b";
  return (
    <group position={[position[0], position[1], height / 2]}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[radius, radius, height, segments]} />
        <meshStandardMaterial
          color={active ? glowColor : "#3d3520"}
          emissive={active ? glowColor : "#000000"}
          emissiveIntensity={active ? (isBig ? 0.9 : 0.5) : 0}
          transparent={!active}
          opacity={active ? 1 : 0.25}
        />
      </mesh>
      {active && isBig && <pointLight color={glowColor} intensity={0.45} distance={4} />}
    </group>
  );
}

/* ── Car ──────────────────────────────────────────────────────────── */
function Car({ pos, yaw, color }) {
  return (
    <group position={pos} rotation={[0, 0, yaw]}>
      {/* Lower chassis */}
      <mesh castShadow position={[0, 0, -0.06]}>
        <boxGeometry args={[2.0, 1.2, 0.4]} />
        <meshStandardMaterial color={color} metalness={0.55} roughness={0.3} />
      </mesh>
      {/* Cabin */}
      <mesh castShadow position={[-0.2, 0, 0.22]}>
        <boxGeometry args={[1.0, 1.0, 0.3]} />
        <meshStandardMaterial color={color} metalness={0.5} roughness={0.35} />
      </mesh>
      {/* Windshield */}
      <mesh position={[0.25, 0, 0.22]}>
        <boxGeometry args={[0.08, 0.85, 0.28]} />
        <meshStandardMaterial color="#0a1a2e" metalness={0.9} roughness={0.1} transparent opacity={0.85} />
      </mesh>
      {/* Front bumper */}
      <mesh position={[0.92, 0, -0.16]}>
        <boxGeometry args={[0.2, 1.1, 0.22]} />
        <meshStandardMaterial color="#222222" metalness={0.4} roughness={0.6} />
      </mesh>
      {/* Spoiler */}
      <mesh position={[-0.9, 0, 0.3]}>
        <boxGeometry args={[0.06, 1.0, 0.04]} />
        <meshStandardMaterial color={color} metalness={0.7} roughness={0.2} />
      </mesh>
      {/* Spoiler supports */}
      <mesh position={[-0.9, 0.35, 0.15]}><boxGeometry args={[0.04, 0.04, 0.25]} /><meshStandardMaterial color="#333" /></mesh>
      <mesh position={[-0.9, -0.35, 0.15]}><boxGeometry args={[0.04, 0.04, 0.25]} /><meshStandardMaterial color="#333" /></mesh>
      {/* Wheels */}
      {[[0.7, 0.62, -0.34], [0.7, -0.62, -0.34], [-0.7, 0.62, -0.34], [-0.7, -0.62, -0.34]].map((p, i) => (
        <group key={i} position={p}>
          <mesh rotation={[0, Math.PI / 2, 0]} castShadow>
            <cylinderGeometry args={[0.22, 0.22, 0.18, 16]} />
            <meshStandardMaterial color="#111" roughness={0.9} />
          </mesh>
          <mesh rotation={[0, Math.PI / 2, 0]}>
            <cylinderGeometry args={[0.13, 0.13, 0.19, 6]} />
            <meshStandardMaterial color="#555" metalness={0.9} roughness={0.2} />
          </mesh>
        </group>
      ))}
    </group>
  );
}

/* ── Ball ─────────────────────────────────────────────────────────── */
function Ball({ pos }) {
  return (
    <group position={pos}>
      <mesh castShadow>
        <sphereGeometry args={[0.45, 32, 32]} />
        <meshStandardMaterial color="#f5a623" emissive="#f5a623" emissiveIntensity={0.35} metalness={0.15} roughness={0.25} />
      </mesh>
      {/* Panel pattern wireframe */}
      <mesh>
        <sphereGeometry args={[0.46, 12, 8]} />
        <meshBasicMaterial color="#222" wireframe transparent opacity={0.25} />
      </mesh>
      <pointLight color="#f5a623" intensity={0.55} distance={6} />
    </group>
  );
}

/* ── Main Arena ──────────────────────────────────────────────────── */
export default function Arena({ state }) {
  const bot1 = state?.bot1 ?? [-8, 0, 0.45, 0];
  const bot2 = state?.bot2 ?? [8, 0, 0.45, Math.PI];
  const ball = state?.ball ?? [0, 0, 1.5];
  const padStates = state?.boost_pads ?? Array(BIG_PADS.length + SMALL_PADS.length).fill(true);

  return (
    <Canvas
      shadows
      gl={{ antialias: true }}
      onCreated={({ camera }) => { camera.up.set(0, 0, 1); }}
    >
      <color attach="background" args={["#0a1929"]} />
      <fog attach="fog" args={["#0a1929", 45, 85]} />

      <PerspectiveCamera makeDefault position={[0, -32, 16]} fov={48} />
      <OrbitControls target={[0, 0, 1.5]} maxPolarAngle={Math.PI * 0.48} />

      {/* Stadium lighting */}
      <ambientLight intensity={0.3} />
      <directionalLight position={[12, -8, 22]} intensity={1.1} castShadow shadow-mapSize-width={1024} shadow-mapSize-height={1024} />
      <directionalLight position={[-10, 6, 18]} intensity={0.5} />
      <pointLight position={[0, 0, H - 0.5]} intensity={0.4} color="#e0e7ff" />
      <pointLight position={[-L / 4, 0, H - 0.5]} intensity={0.25} color="#bfdbfe" />
      <pointLight position={[L / 4, 0, H - 0.5]} intensity={0.25} color="#fed7aa" />

      <Field />
      <Walls />
      <Goal side="blue" />
      <Goal side="orange" />

      {/* Boost pads */}
      {BIG_PADS.map(([x, y], i) => (
        <BoostPadMesh key={`b${i}`} position={[x, y]} isBig active={padStates[i] ?? true} />
      ))}
      {SMALL_PADS.map(([x, y], i) => (
        <BoostPadMesh key={`s${i}`} position={[x, y]} isBig={false} active={padStates[BIG_PADS.length + i] ?? true} />
      ))}

      <Car pos={[bot1[0], bot1[1], bot1[2]]} yaw={bot1[3]} color="#38bdf8" />
      <Car pos={[bot2[0], bot2[1], bot2[2]]} yaw={bot2[3]} color="#fb7185" />
      <Ball pos={[ball[0], ball[1], ball[2]]} />
    </Canvas>
  );
}
