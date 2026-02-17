import { useState } from "react";
import { ALL_EXPERIMENTS } from "../../api/types";
import type { BenchmarkRequest } from "../../api/types";
import { Play } from "lucide-react";

interface Props {
  onSubmit: (request: BenchmarkRequest) => void;
  disabled?: boolean;
}

export default function BenchmarkForm({ onSubmit, disabled }: Props) {
  const [experiments, setExperiments] = useState<string[]>(["all"]);
  const [n, setN] = useState(1000);
  const [seed, setSeed] = useState(42);
  const [noPerf, setNoPerf] = useState(false);
  const [perfRounds, setPerfRounds] = useState(5);
  const [ciMode, setCiMode] = useState(false);
  const [notes, setNotes] = useState("");

  const toggleExperiment = (exp: string) => {
    if (exp === "all") {
      setExperiments(["all"]);
      return;
    }
    setExperiments((prev) => {
      const without = prev.filter((e) => e !== "all" && e !== exp);
      if (prev.includes(exp)) return without.length ? without : ["all"];
      return [...without, exp];
    });
  };

  return (
    <div className="space-y-5 rounded-xl border border-gray-800 bg-gray-900/60 p-6">
      {/* Experiments */}
      <div>
        <label className="block text-xs font-medium uppercase text-gray-400 mb-2">
          Experiments
        </label>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => toggleExperiment("all")}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
              experiments.includes("all")
                ? "bg-orange-600 text-white"
                : "bg-gray-800 text-gray-300 hover:bg-gray-700"
            }`}
          >
            all
          </button>
          {ALL_EXPERIMENTS.map((exp) => (
            <button
              key={exp}
              type="button"
              onClick={() => toggleExperiment(exp)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                experiments.includes(exp)
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-300 hover:bg-gray-700"
              }`}
            >
              {exp}
            </button>
          ))}
        </div>
      </div>

      {/* N, Seed, Perf rounds */}
      <div className="grid grid-cols-4 gap-4">
        <div>
          <label className="block text-xs font-medium uppercase text-gray-400 mb-1">
            N (cases)
          </label>
          <input
            type="number"
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            min={10}
            max={100000}
            className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white"
          />
        </div>
        <div>
          <label className="block text-xs font-medium uppercase text-gray-400 mb-1">
            Seed
          </label>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white"
          />
        </div>
        <div>
          <label className="block text-xs font-medium uppercase text-gray-400 mb-1">
            Perf Rounds
          </label>
          <input
            type="number"
            value={perfRounds}
            onChange={(e) => setPerfRounds(Number(e.target.value))}
            min={1}
            max={20}
            disabled={noPerf}
            className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white disabled:opacity-50"
          />
        </div>
        <div className="flex flex-col justify-end gap-2">
          <label className="flex items-center gap-2 text-sm text-gray-300">
            <input
              type="checkbox"
              checked={noPerf}
              onChange={(e) => setNoPerf(e.target.checked)}
              className="rounded border-gray-600"
            />
            Skip perf
          </label>
          <label className="flex items-center gap-2 text-sm text-gray-300">
            <input
              type="checkbox"
              checked={ciMode}
              onChange={(e) => setCiMode(e.target.checked)}
              className="rounded border-gray-600"
            />
            CI mode
          </label>
        </div>
      </div>

      {/* Notes */}
      <div>
        <label className="block text-xs font-medium uppercase text-gray-400 mb-1">
          Notes (optional)
        </label>
        <input
          type="text"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Why this run exists..."
          className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white placeholder-gray-600"
        />
      </div>

      {/* Submit */}
      <button
        type="button"
        disabled={disabled}
        onClick={() =>
          onSubmit({
            experiments,
            n,
            seed,
            no_perf: noPerf,
            perf_rounds: perfRounds,
            ci_mode: ciMode,
            notes,
          })
        }
        className="flex items-center gap-2 rounded-lg bg-orange-600 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <Play className="h-4 w-4" />
        Run Benchmark
      </button>
    </div>
  );
}
