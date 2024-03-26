# Stage 1: Build the Angular application
FROM node:20 as build-stage


# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json (if available)
COPY package*.json ./

# Install Angular CLI globally and project dependencies
RUN npm install 
# -g @angular/cli && npm install

# Copy the rest of the application files to the container
COPY . .


# List the content of the dist/frontend directory
RUN ls -l ./dist/frontend

# Build the app for production with AOT compilation
RUN npm run build -- --output-path=./dist/frontend
# RUN ng build --configuration=production


# Using Nginx to serve the application
FROM nginx:alpine

# Copy built assets from builder stage to nginx public folder
# Here you need to replace 'my-app' with the actual path to your build output directory inside the dist/ folder.
COPY --from=build-stage /app/dist/frontend/ /usr/share/nginx/html

# Copy custom Nginx configuration
COPY default.conf /etc/nginx/conf.d/default.conf

# Expose port 80 to the outside once the container has launched
EXPOSE 80

# Start Nginx and serve the application
CMD ["nginx", "-g", "daemon off;"]
