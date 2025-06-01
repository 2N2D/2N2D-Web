import type {Metadata} from "next";
import {Geist, Geist_Mono} from "next/font/google";
import "@/lib/fontawesome/css/fa.css";
import "./globals.css";
import SideBar from "@/components/SideBar";
import Chat from "@/components/chat";

const geistSans = Geist({
    variable: "--font-geist-sans",
    subsets: ["latin"],
});

const geistMono = Geist_Mono({
    variable: "--font-geist-mono",
    subsets: ["latin"],
});

export const metadata: Metadata = {
    title: "2N2D",
    description: "Neural Network Development Dashboard",
};

export default function RootLayout({
                                       children,
                                   }: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
        <head>
            <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script type="text/javascript" src="/eel.js"></script>

            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/FontLoader.min.js"></script>
            <script
                src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/geometries/TextGeometry.min.js"></script>
        </head>
        <body
            className={`${geistSans.variable} ${geistMono.variable} antialiased`}
        >
        <SideBar/>
        <Chat/>
        <div className="container">{children}</div>
        <footer>
            <p>Neural Network Development Dashboard</p>
        </footer>
        </body>
        </html>
    );
}
