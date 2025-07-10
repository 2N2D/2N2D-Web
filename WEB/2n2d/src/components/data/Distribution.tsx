"use client"

import Plot from "react-plotly.js";

type Distribution = {
    values: number[];
    bins: number[];
    counts: number[];
    quartiles: number[];
    mean: number;
    stddev: number;
};

type DistributionData = {
    [key: string]: Distribution;
};

type Props = {
    data: DistributionData;

};

export default function PlotDistribution({data}: Props) {
    const keys = Object.keys(data);

    return (
        <div className="flex flex-row w-full overflow-x-auto gap-[0.5rem] h-[600px] align-center items-center">
            {keys.map((key) => {
                const dist = data[key];
                return (
                    <div className={"p-[0.5rem] w-full h-full flex items-center"} key={key}>
                        <Plot

                            data={[
                                {
                                    x: dist.values,
                                    type: 'histogram',
                                    name: key,
                                    marker: {color: '#4f46e5'},
                                },
                            ]}
                            layout={{
                                paper_bgcolor: 'transparent',
                                plot_bgcolor: 'transparent',
                                title: {text: `${key} Distribution`, font: {color: "white"}},
                                xaxis: {title: {text: key, font: {color: "white"}}, tickfont: {color: "white"}},
                                yaxis: {title: {text: 'Frequency', font: {color: "white"}}, tickfont: {color: "white"}},
                                bargap: 0.05,
                                margin: {t: 40, l: 40, r: 30, b: 40},

                            }}
                            useResizeHandler
                            style={{width: '60rem', height: '450px'}}
                            config={{
                                responsive: true,
                                displayModeBar: false,
                                scrollZoom: false,
                                staticPlot: false,
                            }}
                        />
                    </div>
                );
            })}
        </div>
    );
}
