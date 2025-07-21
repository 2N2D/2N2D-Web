"use client";

import React from "react";
import {motion} from "framer-motion";
import ParticleNetwork from "@/components/visual/particleNetwork";
import "./homeStyles.css";

const containerVariants = {
    hidden: {opacity: 0, y: 30},
    visible: {opacity: 1, y: 0, transition: {duration: 0.6}},
};

const staggerVariants = {
    visible: {
        transition: {
            staggerChildren: 0.15,
        },
    },
};

const cardVariants = {
    hidden: {opacity: 0, y: 20},
    visible: {opacity: 1, y: 0},
};

export default function Home() {
    return (
        <motion.main
            className="pageHome"
            initial="hidden"
            animate="visible"
            variants={containerVariants}
        >
            <ParticleNetwork/>
            <motion.div className="heroContainer" variants={containerVariants}>
                <div className="logo">
                    <img src={"logo2n2d.svg"} alt="logo"/>
                    <h1>
                        <b>N</b>eural <b>N</b>etwork <b>D</b>evelopment <b>D</b>ashboard
                    </h1>
                </div>

                <motion.div
                    className="area hero"
                    variants={containerVariants}
                    viewport={{once: true}}
                >
                    <h2 className="subtitle">
                        The one stop shop for your neural network needs
                    </h2>
                    <p>
                        Just drag and drop your files and get going—no setup, no fuss.
                        Whether you're training a simple model or fine-tuning a massive
                        transformer, 2N2D adapts to your workflow. All you need to get
                        started is an account.
                    </p>
                    <motion.a
                        href="/signup"
                        className="ctaButton"
                        whileHover={{scale: 1.05}}
                        whileTap={{scale: 0.95}}
                    >
                        Join now <i className="fa-solid fa-arrow-up-right-from-square"></i>
                    </motion.a>
                </motion.div>
            </motion.div>

            <motion.section
                className="featuresSection"
                initial="hidden"
                whileInView="visible"
                viewport={{once: true, amount: 0.3}}
                variants={staggerVariants}
            >
                <h1 className="title">Features</h1>
                <div className="featuresGrid">
                    {[
                        {
                            title: "Visualize",
                            description:
                                "Easily visualize the structure and nodes of your neural network, in an easy and simple to understand format",
                            icon: "fa-chart-network",
                        },
                        {
                            title: "Analyze",
                            description: "Easily interpret training and output data from your models.",
                            icon: "fa-chart-simple",
                        },
                        {
                            title: "Optimize",
                            description:
                                "Optimize your neural network with a better architecture to improve its speed and outputs.",
                            icon: "fa-rabbit-running",
                        },
                        {
                            title: "Ask",
                            description:
                                "Easily visualize the structure and nodes of your neural network, in an easy and simple to understand format",
                            icon: "fa-comments",
                        },
                    ].map(({title, description, icon}, i) => (
                        <motion.div
                            className="card"
                            key={i}
                            variants={cardVariants}
                            whileHover={{scale: 1.05, boxShadow: "0 8px 20px rgba(0,0,0,0.15)"}}
                        >
                            <div>
                                <h2 className="subtitle">{title}</h2>
                                <p>{description}</p>
                            </div>
                            <i className={`fa-solid ${icon} featureIcon`}></i>
                        </motion.div>
                    ))}
                </div>
            </motion.section>


        </motion.main>
    );
}
