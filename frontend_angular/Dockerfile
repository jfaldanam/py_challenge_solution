# Stage 1: Build the Angular application
FROM node:24-alpine AS build

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy the rest of the application
COPY . .

# Build the application for production with docker configuration
RUN npm run build -- --configuration prod

# Stage 2: Serve the application with nginx
FROM nginx:alpine

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy the built application from the build stage
COPY --from=build /app/dist/py_challenge_frontend_angular/browser /usr/share/nginx/html

# Create nginx configuration for port 8780 and SPA routing
RUN echo 'server { \
    listen 8780; \
    server_name localhost; \
    root /usr/share/nginx/html; \
    index index.html; \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

# Expose port 8780
EXPOSE 8780

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
