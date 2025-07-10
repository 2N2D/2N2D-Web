"use client";

import React, {useState, useEffect, Suspense} from "react";
import "./styles.css";
import CSVUploader from "@/components/fileUploadElements/CSVUploader";
import {deleteCsv} from "@/lib/sessionHandling/sessionUpdater";
import {motion} from "framer-motion";
import DataTable from "@/components/data/DataTable"

function Data() {
    const [missed, setMissed] = useState<number>();
    const [result, setResult] = useState<any>(null);
    const [selectedView, setSelectedView] = useState<number>(0);
    let Heatmap, MissingDataHeatmap = null;
    if (selectedView == 1)
        Heatmap = React.lazy(() => import("@/components/data/HeatMap"))
    if (selectedView == 2)
        MissingDataHeatmap = React.lazy(() => import("@/components/data/MissingValues"))

    function handleNewData() {
        const data = sessionStorage.getItem("csvData");
        if (!data || data.length < 4) return;
        const _result = JSON.parse(data);
        setResult(_result);
        console.log(_result)
        let missing = 0;
        for (let key in _result.summary.missingValues) {
            missing += _result.summary.missingValues[key];
        }
        setMissed(missing);
    }

    async function clearData() {
        const curSesId = sessionStorage.getItem("currentSessionId");
        if (!curSesId) return;
        await deleteCsv(parseInt(curSesId));

        setResult(null);
        sessionStorage.removeItem("csvData");
    }


    useEffect(() => {
        handleNewData();
    }, [])

    return (
        <motion.div className="pageData" transition={{delay: 0.4, duration: 0.2, ease: "easeOut"}}
                    initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}}>
            <div className={"flex gap-[0.1rem] w-full h-auto"}>
                <div className={"titleArea h-full"}>
                    <h1 className={"dataTitle title"}>CSV tools</h1>
                </div>
                <div className={"flex flex-col w-[50%] gap-[0.1rem]"}>
                    <div className="area">
                        <div className={"dataArea w-full"}>
                            <h1 className={"subtitle"}>Add dataset:</h1>
                            <CSVUploader callBack={handleNewData}/>
                            <button className={"deleteButton"} onClick={clearData}>
                                Clear Data <i className="fa-solid fa-trash-xmark"></i>
                            </button>
                        </div>
                    </div>
                    <div className={"flex gap-[0.1rem] h-full"}>
                        <div className="area w-full gap-[0.55rem] h-full">
                            <h3 className={"subtitle text-[var(--warning-color)]"}>Warnings</h3>
                            <div
                                className={"flex flex-col h-[11rem] border-1 border-[var(--border-color)] rounded-[0.4rem] overflow-y-auto p-[0.1rem]"}>
                                {
                                    result && result.results && result.results.encoding_feasibility.warnings.length > 0 ?
                                        result.results.encoding_feasibility.warnings.length.map((warn: string, i: number) =>
                                            <div key={i} className={"warningItem"}>
                                                <p>{warn}</p>
                                            </div>) :
                                        <div className={"warningItem"}>
                                            <p>No warnings</p>
                                        </div>
                                }
                            </div>
                        </div>
                        <div className="area w-full gap-[0.55rem] h-full">
                            <h3 className={"subtitle text-[var(--primary-color)]"}>Recommendations</h3>
                            <div
                                className={"flex flex-col h-[11rem] border-1 border-[var(--border-color)] rounded-[0.4rem] overflow-y-auto p-[0.1rem]"}>
                                {
                                    result && result.results && result.results.encoding_feasibility.recommendations.length > 0 ?
                                        result.results.encoding_feasibility.recommendations.length.map((rec: string, i: number) =>
                                            <div className={"warningItem"} key={i}>
                                                <p>{rec}</p>
                                            </div>) :
                                        <div className={"warningItem"}>
                                            <p>No recommendations</p>
                                        </div>
                                }
                            </div>
                        </div>
                    </div>

                </div>
                <div className={"flex flex-col w-[50%] gap-[0.1rem]"}>
                    <div className="area w-full gap-[1rem]">
                        <h3 className={"subtitle"}>Encoding Feasibility</h3>
                        <div className="dataSum">
                            <div className="info">
                                <h1>One hot</h1>
                                <h2>{result == null ? "-" : result.results.encoding_feasibility.is_safe_for_onehot ? "Safe" : "Unsafe"}</h2>
                            </div>
                            <div className="info">
                                <h1>Cur. Memory</h1>
                                <h2>{result == null ? "-" : result.results.encoding_feasibility.memory_estimate.current_memory_mb + "mb"}</h2>
                            </div>
                            <div className="info">
                                <h1>Est. Memory</h1>
                                <h2>{result == null ? "-" : result.results.encoding_feasibility.memory_estimate.estimated_memory_mb + "mb"}</h2>
                            </div>
                            <div className="info">
                                <h1>Overall</h1>
                                <h2>{result == null ? "-" : result.results.encoding_feasibility.overall_recommendation}</h2>
                            </div>

                        </div>
                    </div>
                    <div className="area w-full gap-[1rem]">
                        <h3 className={"subtitle"}>Dataset Overview</h3>
                        <div className="dataSum">
                            <div className="info">
                                <h1>File</h1>
                                <h2>{result == null ? "No file uploaded" : result.summary.filename}</h2>
                            </div>
                            <div className="info">
                                <h1>Rows</h1>
                                <h2>{result == null ? "-" : result.summary.rows}</h2>
                            </div>
                            <div className="info">
                                <h1>Columns</h1>
                                <h2>{result == null ? "-" : result.summary.columns}</h2>
                            </div>
                            <div className="info">
                                <h1>Missing values</h1>
                                <h2>{result == null ? "-" : missed}</h2>
                            </div>

                        </div>
                    </div>

                </div>
            </div>

            <div className="area tableArea">
                <Suspense fallback={"Loading table...."}>
                    <DataTable result={result}/>
                </Suspense>
            </div>
            <div className={"area"}>
                <div className={"viewButtons"}>
                    <button onClick={() => {
                        setSelectedView(0)
                    }} style={selectedView == 0 ? {
                        backgroundColor: "var(--primary-color)",
                        color: "var(--card-background)"
                    } : {}}><i
                        className="fa-solid fa-binary-lock"></i> Encoding Info
                    </button>
                    <button onClick={() => {
                        setSelectedView(1)
                    }} style={selectedView == 1 ? {
                        backgroundColor: "var(--primary-color)",
                        color: "var(--card-background)"
                    } : {}}><i className="fa-solid fa-hashtag"></i> Correlation Matrix
                    </button>
                    <button onClick={() => {
                        setSelectedView(2)
                    }} style={selectedView == 2 ? {
                        backgroundColor: "var(--primary-color)",
                        color: "var(--card-background)"
                    } : {}}><i
                        className="fa-solid fa-value-absolute"></i> Missing Values Heatmap
                    </button>
                </div>
                {
                    selectedView == 1 && result && result.results && Heatmap ?
                        <motion.div transition={{delay: 0.4, duration: 0.2, ease: "easeOut"}}
                                    initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}}
                                    className={"area justify-center flex items-center"}>
                            <h1 className={"subtitle"}>Correlation Matrix</h1>
                            <Suspense fallback={
                                <motion.div transition={{delay: 0.4, duration: 0.2, ease: "easeOut"}}
                                            initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}}
                                            className={"flex flex-col gap-[1rem] items-center justify-center h-[600px]"}>
                                    <div className={"spinner"}/>
                                    <h1><b>Loading</b></h1>
                                </motion.div>}>
                                <motion.div transition={{delay: 0.4, duration: 0.2, ease: "easeOut"}}
                                            initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}}
                                            className={"h-full w-full"}>
                                    <Heatmap matrix={result.results.visualization_data.correlation_matrix}/>
                                </motion.div>
                            </Suspense>
                        </motion.div> : ""
                }
                {
                    selectedView == 2 && result && result.results && MissingDataHeatmap ?
                        <motion.div transition={{delay: 0.4, duration: 0.2, ease: "easeOut"}}
                                    initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}}
                                    className={"area justify-center flex items-center"}>
                            <h1 className={"subtitle"}>Missing Data Heatmap</h1>
                            <Suspense fallback={<motion.div transition={{delay: 0.4, duration: 0.2, ease: "easeOut"}}
                                                            initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}}
                                                            className={"flex flex-col gap-[1rem] items-center justify-center h-[600px]"}>
                                <div className={"spinner"}/>
                                <h1><b>Loading</b></h1>
                            </motion.div>}>
                                <motion.div transition={{delay: 0.4, duration: 0.2, ease: "easeOut"}}
                                            initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}}
                                            className={"h-full w-full"}>
                                    <MissingDataHeatmap
                                        columns={result.results.visualization_data.missing_data_heatmap.columns}
                                        data={result.results.visualization_data.missing_data_heatmap.data}/>
                                </motion.div>
                            </Suspense>
                        </motion.div> : ""
                }
                {
                    selectedView == 0 && result && result.results ?
                        <motion.div className={"area justify-center flex flex-col gap-[1rem] h-[600px]"}
                                    transition={{delay: 0.4, duration: 0.2, ease: "easeOut"}}
                                    initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}}>
                            <h1 className={"subtitle"}>Encoding Info</h1>
                            <div className={"p-[0.5rem] rounded-[0.4rem] flex flex-col gap[0.5rem] w-fit"}
                                 style={{backgroundColor: "var(--background-color)"}}>
                                <h1 className={"bold"}>
                                    Overall
                                    recommendation: <b
                                    style={result.results.encoding_feasibility.overall_recommendation == "safe" ? {color: "var(--success-color)"} : {color: "var(--error-color)"}}>{result.results.encoding_feasibility.overall_recommendation == "safe" ? "Safe" : result.results.encoding_feasibility.overall_recommendation}</b>
                                </h1>
                                <h1 className={"bold"}>Onehot
                                    encoding: <b
                                        style={result.results.encoding_feasibility.is_safe_for_onehot ? {color: "var(--success-color)"} : {color: "var(--error-color)"}}>{result.results.encoding_feasibility.is_safe_for_onehot ? "Safe" : "Not safe"}</b>
                                </h1>
                            </div>
                            <div className={"flex justify-evenly align-center p-[0.5rem] rounded-[0.4rem]"}
                                 style={{backgroundColor: "var(--background-color)"}}>
                                <div className={"flex flex-col p-[1rem] rounded-[0.4rem] h-[20rem] w-[20rem]"}
                                     style={{backgroundColor: "var(--card-background)"}}>
                                    <h1 className={"font-bold text-[1.5rem] mb-[0.5rem]"}>Column types:</h1>
                                    <div className={"flex gap-[0.5rem] items-center justify-center"}>
                                        <div>
                                        </div>
                                    </div>
                                </div>
                                <div className={"flex flex-col p-[1rem] rounded-[0.4rem] h-[20rem] w-[20rem]"}
                                     style={{backgroundColor: "var(--card-background)"}}>
                                    <h1 className={"font-bold text-[1.5rem] mb-[0.5rem]"}>Categorical summary:</h1>
                                    <p>Risky for
                                        onehot: {result.results.encoding_feasibility.categorical_summary.risky_for_onehot.toString()}</p>
                                    <p>Safe for
                                        onehot: {result.results.encoding_feasibility.categorical_summary.safe_for_onehot.toString()}</p>
                                    <p>Total
                                        categorical: {result.results.encoding_feasibility.categorical_summary.total_categorical.toString()}</p>
                                </div>
                                <div className={"flex flex-col p-[1rem] rounded-[0.4rem] h-[20rem] w-[20rem]"}
                                     style={{backgroundColor: "var(--card-background)"}}>
                                    <h1 className={"font-bold text-[1.5rem] mb-[0.5rem]"}>Columns estimate:</h1>
                                    <p>Risky for
                                        onehot: {result.results.encoding_feasibility.column_estimate.current_columns.toString()}</p>
                                    <p>Safe for
                                        onehot: {result.results.encoding_feasibility.column_estimate.estimated_final_columns.toString()}</p>
                                    <p>Total
                                        categorical: {result.results.encoding_feasibility.column_estimate.new_columns_added.toString()}</p>
                                </div>
                                <div className={"flex flex-col p-[1rem] rounded-[0.4rem] h-[20rem] w-[20rem]"}
                                     style={{backgroundColor: "var(--card-background)"}}>
                                    <h1 className={"font-bold text-[1.5rem] mb-[0.5rem]"}>High cardinality columns:</h1>
                                    <div>
                                        {result.results.encoding_feasibility.high_cardinality_columns.length > 0 ? result.results.encoding_feasibility.high_cardinality_columns.map((val: string, i: number) =>
                                            <p key={i}>{val}</p>) : <p>No such columns</p>}
                                    </div>
                                </div>
                            </div>
                        </motion.div> : ""
                }
            </div>
        </motion.div>
    );
}

export default Data;
