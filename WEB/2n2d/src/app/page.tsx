"use client"

import Styles from "@/components/SideBar.module.css";
import React from "react";
import "./homeStyles.css"
import ParticleNetwork from "@/components/visual/particleNetwork";

export default function Home() {
    return <div className="pageHome">
        <div className={"heroContainer"}>
            <div className={"logo"}>
                <img
                    src={"logo2n2d.svg"}
                    alt="logo"
                />
                <h1><b>N</b>eural <b>N</b>etwork <b>D</b>evelopment <b>D</b>ashboard</h1>
            </div>
            <div className={"area justify-center flex flex-col gap-[1rem] hero"}>
                <div className={"flex flex-col gap-[1rem] w-full"}>
                    <h2 className={"subtitle"}>The one shop stop for you neural network needs</h2>
                    <p>Just drag and drop your files and get going—no setup, no fuss.
                        Whether you're training a simple model or fine-tuning a massive transformer, 2N2D adapts to your
                        workflow.
                        All you need to get started is an account.</p>
                    <a href={"/signup"}>Join now <i className="fa-solid fa-arrow-up-right-from-square"></i></a>
                </div>
            </div>
        </div>
        <div>

            <h1 className={"title ml-[10rem]"}>Features</h1>
            <div className={"flex gap-[4rem] w-max-full flex-wrap justify-center align-center"}>
                <div className={"card"}>
                    <div className={"flex flex-col gap-[1rem] w-full"}>
                        <h2 className={"subtitle"}>Visualize</h2>
                        <p>Easily visualize the structure and nodes of your neural network, in an easy and simple to
                            understand
                            format</p>
                    </div>
                    <i className="fa-solid fa-chart-network"></i>
                </div>
                <div className={"card"}>
                    <div className={"flex flex-col gap-[1rem] w-full"}>
                        <h2 className={"subtitle"}>Analyze</h2>
                        <p>Easily interpret training and output data from your models. </p>
                    </div>
                    <i className="fa-solid fa-chart-simple"></i>
                </div>
                <div className={"card"}>
                    <div className={"flex flex-col gap-[1rem] w-full"}>
                        <h2 className={"subtitle"}>Optimize</h2>
                        <p>Optimize your neural network with a better architecture to improve its speed and outputs.</p>

                    </div>
                    <i className="fa-solid fa-rabbit-running"></i>
                </div>
                <div className={"card"}>
                    <div className={"flex flex-col gap-[1rem] w-full"}>
                        <h2 className={"subtitle"}>Ask</h2>
                        <p>Easily visualize the structure and nodes of your neural network, in an easy and simple to
                            understand
                            format</p>
                    </div>
                    <i className="fa-solid fa-comments"></i>
                </div>
            </div>
        </div>
        <ParticleNetwork/>

    </div>;
}
