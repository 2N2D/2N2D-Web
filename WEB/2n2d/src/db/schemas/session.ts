import {pgTable, serial, text, json} from "drizzle-orm/pg-core"

export const session = pgTable("session", {
    id: serial("id").primaryKey(),
    tokenHash: text("tokenhash").notNull(),
    userId: text("userid").notNull(),
    onnxName: text("onnxname"),
    onnxUrl: text("onnxurl"),
    csvName: text("csvname"),
    csvUrl: text("csvurl"),
    optimizedFileUrl: text("optimizedfileurl"),
    visResult: json("visresult"),
    csvResult: json("csvresult"),
    optResult: json("optresult"),
    chat: json("chat"),
    name: text("name")
});