import { useEffect, useRef } from "react";

interface Props {
  lines: string[];
  status?: string;
}

export default function LogStream({ lines, status }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines.length]);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-950 overflow-hidden">
      <div className="flex items-center justify-between border-b border-gray-800 px-4 py-2">
        <span className="text-xs font-medium uppercase text-gray-400">
          Benchmark Output
        </span>
        {status && (
          <span
            className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
              status === "running"
                ? "bg-yellow-900/50 text-yellow-300"
                : status === "completed"
                ? "bg-green-900/50 text-green-300"
                : status === "failed"
                ? "bg-red-900/50 text-red-300"
                : "bg-gray-800 text-gray-400"
            }`}
          >
            {status}
          </span>
        )}
      </div>
      <pre className="h-[400px] overflow-y-auto p-4 text-xs leading-5 text-gray-300 font-mono">
        {lines.length === 0 && (
          <span className="text-gray-600">Waiting for output...</span>
        )}
        {lines.map((l, i) => (
          <div key={i}>{l}</div>
        ))}
        <div ref={endRef} />
      </pre>
    </div>
  );
}
