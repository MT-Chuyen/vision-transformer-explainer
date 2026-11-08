/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"],
        serif: ["Iowan Old Style", "Palatino Linotype", "Palatino", "Georgia", "serif"],
      },
    },
  },
  plugins: [],
};
