import React from 'react';
import "./styles.css";

export default function profile() {
    return <main>
        <h1>Profile</h1>
        <form>
            <label>Username</label>
            <input type={"text"}/>
            <input type={"submit"} value={"Save"}/>
        </form>
        <h2>E-Mail:</h2>
        <p></p>
        <button>Logout</button>
    </main>
}