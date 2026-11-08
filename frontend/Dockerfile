# Multi-stage build: compile the Vite/React/TS dashboard, then serve the
# static bundle with nginx (which also reverse-proxies /events, /health, and
# /ws/ to the FastAPI `dashboard` service -- see nginx.conf).
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
