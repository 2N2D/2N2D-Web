import React from 'react';
import "./styles.css";
import ParticleNetwork from "@/components/visual/particleNetwork";

export default function dash() {

    return <main className={"pageDash"}>
        <ParticleNetwork/>
        <div className={"lander"}>
            <div>
                <h1 className={"title"}>Dashboard</h1>
                <h2 className={"subtitle"}>Hello, <b>username!</b></h2>
            </div>
            <img
                src={"logo2n2d.svg"}
                alt="logo"
                className={"logo"}
            />
        </div>

        <div className={"lastSession"}>
            <h1 className={"title"}>Last session:</h1>
            <button>
                <div>
                    <h1 className={"subtitle"}>Session name</h1>
                    <h2><b>ONNX file:</b> model.onnx</h2>
                    <h2><b>Csv file:</b> data.csv</h2>
                </div>
                <i className="fa-solid fa-clock-rotate-left"></i>
            </button>
        </div>

        <div className={"sessionsArea"}>
            <h2 className={"title"}>Sessions:</h2>
            <div className={"sessions"}>
                <button>
                    <h2 className={"subtitle"}>Create new session</h2>
                    <i className="fa-solid fa-square-plus"></i>
                </button>
                <button>
                    <div className={"flex flex-col"}>
                        <h1 className={"subtitle"}>Session name</h1>
                        <h2><b>ONNX file:</b> model.onnx</h2>
                        <h2><b>Csv file:</b> data.csv</h2>
                    </div>
                    <i className="fa-solid fa-diagram-project"></i>
                </button>
                <button>
                    <div className={"flex flex-col"}>
                        <h1 className={"subtitle"}>Session name</h1>
                        <h2><b>ONNX file:</b> model.onnx</h2>
                        <h2><b>Csv file:</b> data.csv</h2>
                    </div>
                    <i className="fa-solid fa-diagram-project"></i>
                </button>
                <button>
                    <div className={"flex flex-col"}>
                        <h1 className={"subtitle"}>Session name</h1>
                        <h2><b>ONNX file:</b> model.onnx</h2>
                        <h2><b>Csv file:</b> data.csv</h2>
                    </div>
                    <i className="fa-solid fa-diagram-project"></i>
                </button>
                <button>
                    <div className={"flex flex-col"}>
                        <h1 className={"subtitle"}>Session name</h1>
                        <h2><b>ONNX file:</b> model.onnx</h2>
                        <h2><b>Csv file:</b> data.csv</h2>
                    </div>
                    <i className="fa-solid fa-diagram-project"></i>
                </button>
                <button>
                    <div className={"flex flex-col"}>
                        <h1 className={"subtitle"}>Session name</h1>
                        <h2><b>ONNX file:</b> model.onnx</h2>
                        <h2><b>Csv file:</b> data.csv</h2>
                    </div>
                    <i className="fa-solid fa-diagram-project"></i>
                </button>
            </div>
        </div>

        <div className={"fade"}>
        </div>
    </main>
}