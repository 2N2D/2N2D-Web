import {pgTable, serial, text} from "drizzle-orm/pg-core"

export const user = pgTable("user", {
    id: serial("id").primaryKey().notNull(),
    uid: text("uid").notNull(),
    email: text("email").notNull(),
    displayName: text("displayname"),
    sessions: text("sessions").array()
})