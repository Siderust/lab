import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js-dist-min";

interface Props {
  x: number[];        // e.g. RA values
  y: number[];        // e.g. Dec values
  z: number[];        // error values
  xLabel?: string;
  yLabel?: string;
  title?: string;
}

/**
 * Sky / parameter heatmap — colored by error.
 * Works best when inputs form a grid.
 */
export default function HeatmapPlot({
  x,
  y,
  z,
  xLabel = "RA (rad)",
  yLabel = "Dec (rad)",
  title = "Error Heatmap",
}: Props) {
  if (x.length === 0) {
    return (
      <div className="flex h-[280px] items-center justify-center rounded-xl border border-gray-800 text-sm text-gray-500">
        No heatmap data available.
      </div>
    );
  }

  const trace: Data = {
    x,
    y,
    mode: "markers",
    type: "scatter",
    marker: {
      color: z,
      colorscale: "Viridis",
      size: 5,
      colorbar: {
        title: { text: "Error", font: { color: "#d1d5db" } },
        tickfont: { color: "#9ca3af" },
      },
    },
  };

  const layout: Partial<Layout> = {
    title: { text: title, font: { color: "#e5e7eb", size: 14 } },
    xaxis: {
      title: { text: xLabel },
      color: "#9ca3af",
      gridcolor: "#374151",
      zerolinecolor: "#374151",
    },
    yaxis: {
      title: { text: yLabel },
      color: "#9ca3af",
      gridcolor: "#374151",
      zerolinecolor: "#374151",
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(17,24,39,0.6)",
    font: { color: "#d1d5db" },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    height: 350,
  };

  return (
    <Plot
      data={[trace]}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      className="w-full"
    />
  );
}
