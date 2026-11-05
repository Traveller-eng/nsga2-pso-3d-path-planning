import { Html, Line, OrbitControls, Grid } from '@react-three/drei';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import type { MutableRefObject } from 'react';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import type { Candidate, Obstacle, ScenarioData, Vec3 } from './types';

export type CameraPreset = 'overview' | 'side' | 'top' | 'goal';

type WorldSceneProps = {
  scenario: ScenarioData;
  selected: Candidate | null;
  candidates: Candidate[];
  previous: Candidate | null;
  currentTime: number;
  playing: boolean;
  showObstacles: boolean;
  showCandidates: boolean;
  showControlPoints: boolean;
  showReference: boolean;
  followVehicle: boolean;
  reduceMotion: boolean;
  quality: 'balanced' | 'high';
  cameraPreset: CameraPreset;
  onSelectCandidate: (index: number) => void;
};

const asThree = ([x, y, z]: Vec3): [number, number, number] => [x, z, y];

function sampleCandidatePosition(candidate: Candidate, currentTime: number, duration: number): THREE.Vector3 {
  if (candidate.positions.length === 0) return new THREE.Vector3();
  if (candidate.positions.length === 1) return new THREE.Vector3(...asThree(candidate.positions[0]));
  const times = candidate.trajectory_times.length === candidate.positions.length
    ? candidate.trajectory_times
    : candidate.positions.map((_, index) => (duration * index) / Math.max(1, candidate.positions.length - 1));
  const clamped = THREE.MathUtils.clamp(currentTime, times[0] ?? 0, times[times.length - 1] ?? duration);
  let index = 0;
  while (index < times.length - 2 && clamped > times[index + 1]) index += 1;
  const t0 = times[index] ?? 0;
  const t1 = times[index + 1] ?? duration;
  const alpha = t1 === t0 ? 0 : (clamped - t0) / (t1 - t0);
  const from = new THREE.Vector3(...asThree(candidate.positions[index]));
  const to = new THREE.Vector3(...asThree(candidate.positions[index + 1]));
  return from.lerp(to, alpha);
}

function trailLength(candidate: Candidate, currentTime: number, duration: number): number {
  if (candidate.positions.length < 2) return candidate.positions.length;
  if (candidate.trajectory_times.length !== candidate.positions.length) {
    return Math.max(2, Math.floor((currentTime / duration) * candidate.positions.length));
  }
  const index = candidate.trajectory_times.findIndex((time) => time > currentTime);
  return Math.max(2, index === -1 ? candidate.positions.length : index + 1);
}

function motionPosition(obstacle: Obstacle, absoluteTime: number): Vec3 {
  const motion = obstacle.motion;
  if (!motion) return obstacle.center ?? [0, 0, 0];
  if (motion.kind === 'piecewise_linear') {
    const { path, path_times: times } = motion;
    if (absoluteTime <= times[0]) return path[0];
    if (absoluteTime >= times[times.length - 1]) return path[path.length - 1];
    for (let index = 0; index < times.length - 1; index += 1) {
      const t0 = times[index];
      const t1 = times[index + 1];
      if (t0 <= absoluteTime && absoluteTime <= t1) {
        const fraction = t1 === t0 ? 0 : (absoluteTime - t0) / (t1 - t0);
        return path[index].map((coordinate, axis) => coordinate + fraction * (path[index + 1][axis] - coordinate)) as Vec3;
      }
    }
    return path[path.length - 1];
  }
  if (motion.kind === 'linear') {
    return motion.initial_center.map((coordinate, axis) => coordinate + motion.velocity[axis] * absoluteTime) as Vec3;
  }
  return [0, 0, 0];
}

function motionBounds(obstacle: Obstacle, absoluteTime: number): { lower: Vec3; upper: Vec3 } {
  const motion = obstacle.motion;
  if (motion?.kind === 'linear_aabb') {
    return {
      lower: motion.initial_lower.map((value, axis) => value + motion.velocity[axis] * absoluteTime) as Vec3,
      upper: motion.initial_upper.map((value, axis) => value + motion.velocity[axis] * absoluteTime) as Vec3,
    };
  }
  return { lower: obstacle.lower ?? [0, 0, 0], upper: obstacle.upper ?? [1, 1, 1] };
}

function ObstacleMesh({ obstacle, absoluteTime, dSafe }: { obstacle: Obstacle; absoluteTime: number; dSafe: number }) {
  const objectRef = useRef<THREE.Group>(null);
  const sphereGeometry = useMemo(() => new THREE.SphereGeometry(obstacle.radius ?? 1, 40, 28), [obstacle.radius]);
  const shellGeometry = useMemo(() => new THREE.SphereGeometry((obstacle.radius ?? 1) + dSafe, 40, 28), [obstacle.radius, dSafe]);
  const boxBounds = motionBounds(obstacle, absoluteTime);
  const spherePosition = motionPosition(obstacle, absoluteTime);
  const boxSize: Vec3 = [
    boxBounds.upper[0] - boxBounds.lower[0],
    boxBounds.upper[1] - boxBounds.lower[1],
    boxBounds.upper[2] - boxBounds.lower[2],
  ];
  const boxCenter: Vec3 = [
    (boxBounds.upper[0] + boxBounds.lower[0]) / 2,
    (boxBounds.upper[1] + boxBounds.lower[1]) / 2,
    (boxBounds.upper[2] + boxBounds.lower[2]) / 2,
  ];
  useFrame(() => {
    if (!objectRef.current) return;
    const point = obstacle.kind === 'sphere' ? spherePosition : boxCenter;
    objectRef.current.position.set(...asThree(point));
  });

  if (obstacle.kind === 'sphere') {
    return (
      <group ref={objectRef} position={asThree(spherePosition)}>
        <mesh geometry={shellGeometry}>
          <meshBasicMaterial color="#dc9348" transparent opacity={0.075} wireframe depthWrite={false} />
        </mesh>
        <mesh geometry={sphereGeometry} castShadow receiveShadow>
          <meshStandardMaterial color="#4b3428" roughness={0.42} metalness={0.36} emissive="#2f1c13" emissiveIntensity={0.16} transparent opacity={0.72} />
        </mesh>
        <mesh geometry={sphereGeometry} scale={1.004}>
          <meshBasicMaterial color="#dfa36b" wireframe transparent opacity={0.13} depthWrite={false} />
        </mesh>
      </group>
    );
  }

  return (
    <group ref={objectRef} position={asThree(boxCenter)}>
      <mesh castShadow receiveShadow>
        <boxGeometry args={asThree(boxSize)} />
        <meshStandardMaterial color="#554032" roughness={0.48} metalness={0.32} transparent opacity={0.58} />
      </mesh>
      <mesh>
        <boxGeometry args={asThree(boxSize.map((value) => value + 0.025) as Vec3)} />
        <meshBasicMaterial color="#d79a53" wireframe transparent opacity={0.42} />
      </mesh>
    </group>
  );
}

function Vehicle({ candidate, currentTime, duration }: { candidate: Candidate; currentTime: number; duration: number }) {
  const vehicleRef = useRef<THREE.Group>(null);
  const rotors = useRef<Array<THREE.Mesh | null>>([]);
  const initialPosition = sampleCandidatePosition(candidate, currentTime, duration);

  useFrame(({ clock }) => {
    if (!vehicleRef.current || candidate.positions.length < 2) return;
    const point = sampleCandidatePosition(candidate, currentTime, duration);
    vehicleRef.current.position.copy(point);
    const nextPoint = sampleCandidatePosition(candidate, currentTime + 0.04, duration);
    const direction = nextPoint.sub(point).normalize();
    if (direction.lengthSq() > 0.001) {
      vehicleRef.current.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), direction);
    }

    rotors.current.forEach((mesh, rotorIndex) => {
      if (!mesh) return;
      mesh.rotation.z = clock.elapsedTime * 18 + rotorIndex * 0.8;
    });
  });

  const rotorOffsets = [
    [-0.22, 0.18, 0],
    [0.22, 0.18, 0],
    [-0.22, -0.18, 0],
    [0.22, -0.18, 0],
  ] as const;

  return (
    <group ref={vehicleRef} position={initialPosition}>
      <mesh castShadow receiveShadow position={[0, 0, 0.05]}>
        <boxGeometry args={[0.34, 0.1, 0.22]} />
        <meshStandardMaterial color={candidate.is_feasible ? '#79e8d5' : '#f1b16c'} emissive={candidate.is_feasible ? '#1e645a' : '#7b3d07'} emissiveIntensity={0.72} metalness={0.45} roughness={0.28} />
      </mesh>
      <mesh castShadow receiveShadow position={[0, 0, -0.06]}>
        <boxGeometry args={[0.18, 0.18, 0.08]} />
        <meshStandardMaterial color="#d5e8f2" emissive="#111a22" metalness={0.32} roughness={0.2} />
      </mesh>
      {rotorOffsets.map(([x, y, z], index) => (
        <group key={`${x}-${y}`} position={[x, y, z]}>
          <mesh ref={(node) => { rotors.current[index] = node; }} rotation={[Math.PI / 2, 0, 0]} castShadow>
            <cylinderGeometry args={[0.05, 0.05, 0.025, 12]} />
            <meshStandardMaterial color="#95a7b4" metalness={0.75} roughness={0.35} />
          </mesh>
          <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0.015]}>
            <boxGeometry args={[0.11, 0.02, 0.02]} />
            <meshStandardMaterial color="#cddbe1" emissive="#8ed7ea" emissiveIntensity={0.12} />
          </mesh>
        </group>
      ))}
      <mesh position={[0, 0, -0.18]}>
        <sphereGeometry args={[0.06, 12, 8]} />
        <meshBasicMaterial color="#dbf7ff" />
      </mesh>
    </group>
  );
}

function CameraRig({
  preset,
  scenario,
  follow,
  reduceMotion,
  currentTime,
  selected,
  controlsRef,
}: {
  preset: CameraPreset;
  scenario: ScenarioData;
  follow: boolean;
  reduceMotion: boolean;
  currentTime: number;
  selected: Candidate | null;
  controlsRef: MutableRefObject<OrbitControlsImpl | null>;
}) {
  const { camera } = useThree();
  const desiredPosition = useRef(new THREE.Vector3(14, 13, -17));
  const desiredTarget = useRef(new THREE.Vector3(5, 5, 5));
  const currentTarget = useRef(new THREE.Vector3(5, 5, 5));
  useFrame((_, delta) => {
    if (preset === 'overview') desiredPosition.current.set(14, 13, -17);
    if (preset === 'side') desiredPosition.current.set(18, 8.5, -2.4);
    if (preset === 'top') desiredPosition.current.set(5, 21, 5.02);
    if (preset === 'goal') desiredPosition.current.set(13.5, 10.5, 15.5);
    desiredTarget.current.set(5, 5, 5);
    if (follow && selected) {
      desiredTarget.current.copy(sampleCandidatePosition(selected, currentTime, scenario.world.t_max));
    }
    const blend = reduceMotion ? 1 : 1 - Math.exp(-3.7 * delta);
    camera.position.lerp(desiredPosition.current, blend);
    currentTarget.current.lerp(desiredTarget.current, blend);
    if (controlsRef.current) {
      controlsRef.current.target.copy(currentTarget.current);
      controlsRef.current.update();
    }
  });
  return null;
}

function WorldContents(props: WorldSceneProps) {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const { scenario, selected, candidates, previous, currentTime } = props;
  const min = scenario.world.bounds_min;
  const max = scenario.world.bounds_max;
  const center: Vec3 = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const size: Vec3 = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  const frameGeometry = useMemo(
    () => new THREE.EdgesGeometry(new THREE.BoxGeometry(size[0], size[2], size[1])),
    [size[0], size[1], size[2]],
  );
  const framePosition = asThree(center);
  const absoluteTime = scenario.event_time + currentTime;
  const selectedColor = selected?.is_feasible ? '#6ce0d0' : '#efa74f';
  const visibleCandidates = props.showCandidates
    ? candidates.filter((candidate) => candidate.index !== selected?.index && candidate.rank <= 2).slice(0, 16)
    : [];
  const trailCount = selected
    ? trailLength(selected, currentTime, scenario.world.t_max)
    : 0;

  return (
    <>
      <color attach="background" args={['#090e16']} />
      <fog attach="fog" args={['#090e16', 17, 38]} />
      <ambientLight intensity={0.82} color="#a9c2d3" />
      <directionalLight position={[8, 14, 8]} intensity={2.2} color="#b9e4ee" castShadow shadow-mapSize-width={1024} shadow-mapSize-height={1024} />
      <pointLight position={[-2, 7, 4]} intensity={52} distance={24} color="#308ca5" />
      <pointLight position={[11, 3, 8]} intensity={34} distance={18} color="#7656b8" />

      <Grid
        position={[center[0], min[2], center[1]]}
        args={[size[0], size[1]]}
        cellSize={1}
        cellThickness={0.52}
        cellColor="#22303b"
        sectionSize={5}
        sectionThickness={0.95}
        sectionColor="#385361"
        fadeDistance={24}
        fadeStrength={1.3}
        infiniteGrid={false}
      />
      <lineSegments geometry={frameGeometry} position={framePosition}>
        <lineBasicMaterial color="#6c8790" transparent opacity={0.56} />
      </lineSegments>
      <Line points={[[min[0], min[2], min[1]], [max[0], min[2], min[1]]]} color="#dc7755" lineWidth={1.5} transparent opacity={0.62} />
      <Line points={[[min[0], min[2], min[1]], [min[0], max[2], min[1]]]} color="#67b992" lineWidth={1.5} transparent opacity={0.62} />
      <Line points={[[min[0], min[2], min[1]], [min[0], min[2], max[1]]]} color="#6ab8d2" lineWidth={1.5} transparent opacity={0.62} />
      <Html position={[max[0], min[2], min[1]]} className="world-axis-label" center><span>X</span></Html>
      <Html position={[min[0], max[2], min[1]]} className="world-axis-label" center><span>Z</span></Html>
      <Html position={[min[0], min[2], max[1]]} className="world-axis-label" center><span>Y</span></Html>

      {props.showObstacles && scenario.world.obstacles.map((obstacle, index) => (
        <ObstacleMesh key={`${obstacle.kind}-${index}`} obstacle={obstacle} absoluteTime={absoluteTime} dSafe={scenario.world.d_safe} />
      ))}

      <group position={asThree(scenario.world.start)}>
        <mesh castShadow><sphereGeometry args={[0.22, 24, 16]} /><meshStandardMaterial color="#75dfc6" emissive="#267e71" emissiveIntensity={0.38} metalness={0.38} roughness={0.22} /></mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]}><torusGeometry args={[0.39, 0.018, 8, 48]} /><meshBasicMaterial color="#69caba" transparent opacity={0.68} /></mesh>
        <Html position={[0, 0.48, 0]} className="world-label" center><span>ORIGIN <b>00</b></span></Html>
      </group>
      <group position={asThree(scenario.world.goal)}>
        <mesh rotation={[0.25, 0.32, 0]} castShadow><icosahedronGeometry args={[0.25, 1]} /><meshStandardMaterial color="#e8e2d1" emissive="#896d35" emissiveIntensity={0.24} metalness={0.52} roughness={0.22} /></mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]}><torusGeometry args={[0.43, 0.015, 8, 48]} /><meshBasicMaterial color="#d5ad61" transparent opacity={0.75} /></mesh>
        <Html position={[0, 0.55, 0]} className="world-label goal-label" center><span>OBJECTIVE <b>09</b></span></Html>
      </group>

      {props.showReference && scenario.world.start && scenario.world.goal && (
        <Line points={[asThree(scenario.world.start), asThree(scenario.world.goal)]} color="#809099" lineWidth={1} dashed dashSize={0.18} gapSize={0.14} transparent opacity={0.48} />
      )}
      {previous && (
        <Line points={previous.positions.map(asThree)} color="#9176cb" lineWidth={1.3} transparent opacity={0.24} />
      )}
      {visibleCandidates.map((candidate) => (
        <Line
          key={`candidate-${candidate.index}`}
          points={candidate.positions.map(asThree)}
          color={candidate.is_feasible ? '#778c9c' : '#a47448'}
          lineWidth={1}
          transparent
          opacity={candidate.is_feasible ? 0.23 : 0.13}
          onClick={() => props.onSelectCandidate(candidate.index)}
        />
      ))}
      {selected && (
        <>
          <Line points={selected.positions.map(asThree)} color={selectedColor} lineWidth={10} transparent opacity={0.13} depthTest={false} />
          <Line points={selected.positions.map(asThree)} color={selectedColor} lineWidth={3.8} transparent opacity={1} depthTest={false} />
          {props.showControlPoints && selected.control_points.map((point, index) => (
            <mesh key={`control-${index}`} position={asThree(point)}>
              <sphereGeometry args={[index === 0 || index === selected.control_points.length - 1 ? 0.085 : 0.11, 16, 12]} />
              <meshStandardMaterial color={index === 0 || index === selected.control_points.length - 1 ? '#e7ddc7' : '#69d8ca'} emissive="#1c6a64" emissiveIntensity={0.28} />
            </mesh>
          ))}
          {props.playing || currentTime > 0 ? (
            <Line points={selected.positions.slice(0, trailCount).map(asThree)} color="#e5fffb" lineWidth={2.5} transparent opacity={0.78} depthTest={false} />
          ) : null}
          <Vehicle candidate={selected} currentTime={currentTime} duration={scenario.world.t_max} />
        </>
      )}

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[center[0], min[2] - 0.02, center[1]]} receiveShadow>
        <planeGeometry args={[size[0], size[1]]} />
        <shadowMaterial transparent opacity={0.13} />
      </mesh>
      <OrbitControls ref={controlsRef} makeDefault enableDamping dampingFactor={0.08} minDistance={6} maxDistance={34} maxPolarAngle={Math.PI * 0.49} />
      <CameraRig
        preset={props.cameraPreset}
        scenario={scenario}
        follow={props.followVehicle}
        reduceMotion={props.reduceMotion}
        currentTime={currentTime}
        selected={selected}
        controlsRef={controlsRef}
      />
    </>
  );
}

export default function WorldScene(props: WorldSceneProps) {
  return (
    <Canvas
      shadows
      dpr={props.quality === 'high' ? [1, 1.5] : [1, 1.15]}
      camera={{ position: [14, 13, -17], fov: 42, near: 0.1, far: 100 }}
      gl={{ antialias: true, alpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' }}
      onCreated={({ gl }) => {
        gl.outputColorSpace = THREE.SRGBColorSpace;
        gl.toneMapping = THREE.ACESFilmicToneMapping;
        gl.toneMappingExposure = 1.12;
      }}
    >
      <WorldContents {...props} />
    </Canvas>
  );
}
