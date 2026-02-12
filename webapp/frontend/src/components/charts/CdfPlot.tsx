import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js-dist-min";
import { LIBRARY_COLORS } from "../../api/types";

interface Series {
  library: string;
  values: number[];
}

interface Props {
  series: Series[];
  xLabel?: string;
  title?: string;
}

/**
 * CDF of absolute error — one curve per library.
 * X-axis log scale (error), Y-axis 0–1 (fraction).
 */
export default function CdfPlot({
  series,
  xLabel = "Absolute error",
  title = "CDF of Absolute Error",
}: Props) {
  const traces: Data[] = series.map((s) => {
    const sorted = [...s.values].sort((a, b) => a - b);
    const y = sorted.map((_, i) => (i + 1) / sorted.length);
    return {
      x: sorted,
      y,
      type: "scatter" as const,
      mode: "lines" as const,
      name: s.library,
      line: { color: LIBRARY_COLORS[s.library] ?? "#9ca3af", width: 2 },
    };
  });

  const layout: Partial<Layout> = {
    title: { text: title, font: { color: "#e5e7eb", size: 14 } },
    xaxis: {
      title: { text: xLabel },
      type: "log",
      color: "#9ca3af",
      gridcolor: "#374151",
      zerolinecolor: "#374151",
    },
    yaxis: {
      title: { text: "Cumulative fraction" },
      range: [0, 1.02],
      color: "#9ca3af",
      gridcolor: "#374151",
      zerolinecolor: "#374151",
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(17,24,39,0.6)",
    font: { color: "#d1d5db" },
    legend: { bgcolor: "rgba(0,0,0,0)", font: { color: "#d1d5db" } },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    height: 350,
  };

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      className="w-full"
    />
  );
}
