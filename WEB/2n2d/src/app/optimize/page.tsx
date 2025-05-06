import React from "react"

function Optimize() {
    return (
        <div id="content-optimize" className="tab-pane">
            <div className="optimize-main-container" >
                <h2>Neural Network Optimization</h2>

                <div className="always-visible-content"
>
                    <h3>Architecture Optimization</h3>
                    <p>Find the optimal neural network architecture for your dataset.</p>

                    <div>
                        {/*<div>*/}
                        {/*    <label>Input Features:</label>*/}
                        {/*    <select id="opt-input-features" multiple>*/}
                        {/*        <option disabled>Upload a CSV dataset to see available features</option>*/}
                        {/*    </select>*/}
                        {/*</div>*/}

                        {/*<div>*/}
                        {/*    <label>Target Feature:</label>*/}
                        {/*    <select id="opt-target-feature">*/}
                        {/*        <option value="" disabled selected>Upload a CSV dataset to see available features*/}
                        {/*        </option>*/}
                        {/*    </select>*/}
                        {/*</div>*/}

                        {/*<div>*/}
                        {/*    <label>Maximum Epochs Per*/}
                        {/*        Configuration:</label>*/}
                        {/*    <input type="number" id="opt-max-epochs" value="10"/>*/}
                        {/*</div>*/}

                        <div>
                            <button id="start-optimization">
                                Start Optimization
                            </button>
                        </div>
                    </div>
                </div>

                <div id="optimization-progress">
                    <h3>Optimization Progress</h3>
                    <div>
                        <div id="opt-progress-bar"></div>
                    </div>
                    <div id="opt-progress-text">Starting optimization...</div>
                </div>

                <div id="optimization-results">
                    <h3>Optimization Results</h3>
                    <div>
                        <h4>Best Architecture</h4>
                        <div>
                            <div>
                                <div>Hidden Layers:</div>
                                <div id="best-layers">-</div>
                            </div>
                            <div>
                                <div>Neurons Per Layer:</div>
                                <div id="best-neurons">-</div>
                            </div>
                            <div>
                                <div>Test Loss (MSE):</div>
                                <div id="best-loss">-</div>
                            </div>
                            <div>
                                <div>R² Score:</div>
                                <div id="best-r2">-</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Optimize;