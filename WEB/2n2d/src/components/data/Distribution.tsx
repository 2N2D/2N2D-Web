import Plot from "react-plotly.js";

export default function FeatureDistributions({distribution_data}) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {Object.entries(distribution_data).map(([feature, dist]) => (
                <div key={feature} className="bg-background rounded-xl p-4 shadow">
                    <Plot
                        data={[
                            {
                                x: dist.bins.slice(1),
                                y: dist.counts,
                                type: "bar",
                                marker: {color: "#3b82f6"},
                            },
                        ]}
                        layout={{
                            title: {
                                text: feature,
                                font: {size: 18, color: "#ffffff"},
                            },
                            paper_bgcolor: "transparent",
                            plot_bgcolor: "transparent",
                            xaxis: {
                                title: "Value",
                                tickfont: {color: "#cccccc"},
                                titlefont: {color: "#cccccc"},
                            },
                            yaxis: {
                                title: "Count",
                                tickfont: {color: "#cccccc"},
                                titlefont: {color: "#cccccc"},
                            },
                            margin: {t: 40, b: 50, l: 50, r: 30},
                        }}
                        config={{responsive: true, displayModeBar: false}}
                        style={{width: "100%", height: "300px"}}
                    />
                </div>
            ))}
        </div>
    );
}