import { Canvas } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";

function Bot({ pos, yaw, color }) {
  return (
    <group position={pos} rotation={[0, 0, yaw]}>
      <mesh castShadow>
        <boxGeometry args={[1.8, 1.3, 0.9]} />
        <meshStandardMaterial color={color} metalness={0.15} roughness={0.4} />
      </mesh>
      <mesh position={[0.62, 0.48, -0.44]} rotation={[0, Math.PI / 2, 0]} castShadow>
        <cylinderGeometry args={[0.16, 0.16, 0.16, 16]} />
        <meshStandardMaterial color="#0f172a" />
      </mesh>
      <mesh position={[0.62, -0.48, -0.44]} rotation={[0, Math.PI / 2, 0]} castShadow>
        <cylinderGeometry args={[0.16, 0.16, 0.16, 16]} />
        <meshStandardMaterial color="#0f172a" />
      </mesh>
      <mesh position={[-0.62, 0.48, -0.44]} rotation={[0, Math.PI / 2, 0]} castShadow>
        <cylinderGeometry args={[0.16, 0.16, 0.16, 16]} />
        <meshStandardMaterial color="#0f172a" />
      </mesh>
      <mesh position={[-0.62, -0.48, -0.44]} rotation={[0, Math.PI / 2, 0]} castShadow>
        <cylinderGeometry args={[0.16, 0.16, 0.16, 16]} />
        <meshStandardMaterial color="#0f172a" />
      </mesh>
    </group>
  );
}

function Ball({ pos }) {
  return (
    <mesh position={pos} castShadow>
      <sphereGeometry args={[0.45, 24, 24]} />
      <meshStandardMaterial color="#f59e0b" emissive="#5f3d00" metalness={0.1} roughness={0.3} />
    </mesh>
  );
}

function ArenaShell() {
  const length = 36;
  const width = 24;
  const height = 8;
  const cornerRadius = 3.6;

  return (
    <group>
      <mesh position={[0, 0, -0.05]} receiveShadow>
        <boxGeometry args={[length, width, 0.1]} />
        <meshStandardMaterial color="#1e293b" />
      </mesh>

      <mesh position={[0, width / 2, height / 2]} receiveShadow>
        <boxGeometry args={[length, 0.2, height]} />
        <meshStandardMaterial color="#334155" transparent opacity={0.65} />
      </mesh>
      <mesh position={[0, -width / 2, height / 2]} receiveShadow>
        <boxGeometry args={[length, 0.2, height]} />
        <meshStandardMaterial color="#334155" transparent opacity={0.65} />
      </mesh>
      <mesh position={[length / 2, 0, height / 2]} receiveShadow>
        <boxGeometry args={[0.2, width, height]} />
        <meshStandardMaterial color="#0f766e" transparent opacity={0.65} />
      </mesh>
      <mesh position={[-length / 2, 0, height / 2]} receiveShadow>
        <boxGeometry args={[0.2, width, height]} />
        <meshStandardMaterial color="#7f1d1d" transparent opacity={0.65} />
      </mesh>

      <mesh position={[length / 2 + 0.05, 0, 1.6]}>
        <boxGeometry args={[0.02, 6, 3.2]} />
        <meshStandardMaterial color="#34d399" transparent opacity={0.35} />
      </mesh>
      <mesh position={[-length / 2 - 0.05, 0, 1.6]}>
        <boxGeometry args={[0.02, 6, 3.2]} />
        <meshStandardMaterial color="#f87171" transparent opacity={0.35} />
      </mesh>

      {/* Rounded corner wall segments to match curved arena mechanics */}
      <mesh position={[length / 2 - cornerRadius, width / 2 - cornerRadius, height / 2]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[cornerRadius, cornerRadius, height, 28, 1, false, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#475569" transparent opacity={0.55} />
      </mesh>
      <mesh position={[length / 2 - cornerRadius, -width / 2 + cornerRadius, height / 2]} rotation={[Math.PI / 2, 0, -Math.PI / 2]}>
        <cylinderGeometry args={[cornerRadius, cornerRadius, height, 28, 1, false, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#475569" transparent opacity={0.55} />
      </mesh>
      <mesh position={[-length / 2 + cornerRadius, width / 2 - cornerRadius, height / 2]} rotation={[Math.PI / 2, 0, Math.PI / 2]}>
        <cylinderGeometry args={[cornerRadius, cornerRadius, height, 28, 1, false, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#475569" transparent opacity={0.55} />
      </mesh>
      <mesh position={[-length / 2 + cornerRadius, -width / 2 + cornerRadius, height / 2]} rotation={[Math.PI / 2, 0, Math.PI]}>
        <cylinderGeometry args={[cornerRadius, cornerRadius, height, 28, 1, false, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#475569" transparent opacity={0.55} />
      </mesh>
    </group>
  );
}

export default function Arena({ state }) {
  const bot1 = state?.bot1 ?? [-8, 0, 0.45, 0];
  const bot2 = state?.bot2 ?? [8, 0, 0.45, Math.PI];
  const ball = state?.ball ?? [0, 0, 1.5];

  return (
    <Canvas
      shadows
      gl={{ antialias: true }}
      onCreated={({ camera }) => {
        camera.up.set(0, 0, 1);
      }}
    >
      <color attach="background" args={["#091222"]} />

      <PerspectiveCamera makeDefault position={[0, -28, 14]} fov={50} />
      <OrbitControls target={[0, 0, 2]} maxPolarAngle={Math.PI * 0.48} />

      <ambientLight intensity={0.4} />
      <directionalLight
        position={[10, -6, 20]}
        intensity={1.2}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <pointLight position={[0, 0, 7]} intensity={0.6} color="#60a5fa" />

      <ArenaShell />

      <Bot pos={[bot1[0], bot1[1], bot1[2]]} yaw={bot1[3]} color="#38bdf8" />
      <Bot pos={[bot2[0], bot2[1], bot2[2]]} yaw={bot2[3]} color="#fb7185" />
      <Ball pos={[ball[0], ball[1], ball[2]]} />
    </Canvas>
  );
}
