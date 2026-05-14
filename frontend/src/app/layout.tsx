import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HaqDar | Legal Assistant",
  description: "An AI-powered legal assistant specializing in Pakistani Law.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <main className="app-container">
          {children}
        </main>
      </body>
    </html>
  );
}
