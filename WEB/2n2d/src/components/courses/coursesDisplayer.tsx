"use client"
import React, {useState} from "react"
import MarkDownViewer from "@/components/courses/markDownViewer";
import Style from "./courseDisplayer.module.css"
import {motion} from "framer-motion";

interface Course {
    title: string,
    path: string,
    icon: string
}

const courses: Course[] = [
    {
        title: "Setting up",
        path: "/courses/setup.md",
        icon: "fa-solid fa-gears"
    },
    {
        title: "Building",
        path: "/courses/building.md",
        icon: "fa-solid fa-cubes"
    },
    {
        title: "Training",
        path: "/courses/training.md",
        icon: "fa-solid fa-dumbbell"
    },
    {
        title: "Testing",
        path: "/courses/testing.md",
        icon: "fa-solid fa-vial"
    },
    {
        title: "Improvements",
        path: "/courses/improvements.md",
        icon: "fa-solid fa-arrow-up-right-dots"
    },
    {
        title: "Real-World Data",
        path: "/courses/realdata.md",
        icon: "fa-solid fa-database"
    },
    {
        title: "Cheat Sheet",
        path: "/courses/cheatsheet.md",
        icon: "fa-solid fa-clipboard-list"
    },
    {
        title: "Tips",
        path: "/courses/tips.md",
        icon: "fa-solid fa-lightbulb"
    }
];

export default function CoursesDisplayer() {
    const [open, setOpen] = useState(false);
    const [path, setPath] = useState("");
    const [name, setName] = useState("");

    function loadCourse(path: string, name: string) {
        setPath(path);
        setName(name);
        setOpen(true);
    }

    return (
        <div>
            <motion.div className={Style.cont} initial="hidden" whileInView="visible" variants={{
                hidden: {opacity: 0, rotate: -2},
                visible: {
                    opacity: 1,
                    rotate: 0,
                    transition: {
                        staggerChildren: 0.1,
                    },
                },

            }} viewport={{once: true, amount: 0.3}}>
                {
                    courses.map((course, i) => {
                        return <motion.button key={i} onClick={() => {
                            loadCourse(course.path, course.title)
                        }} variants={{
                            hidden: {opacity: 0, y: 20},
                            visible: {opacity: 1, y: 0},
                        }}
                                              transition={{duration: 0.2}}
                                              whileHover={{
                                                  scale: 1.05,
                                                  rotate: 2,
                                                  backgroundColor: "var(--primary-color)",
                                                  color: "var(--card-background)",
                                                  transition: {duration: 0.2, ease: "easeInOut"},
                                              }}>
                            <h1>{course.title}</h1>
                            <i className={course.icon}/>
                        </motion.button>
                    })
                }
            </motion.div>
            <MarkDownViewer path={path} title={name} open={open} onClose={() => {
                setOpen(false)
                sessionStorage.removeItem("screenContext")
            }}/>
        </div>
    )
}