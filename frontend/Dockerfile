# Using a Node.js base image
FROM node:18

# Set the working directory
WORKDIR /usr/src/app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the application files
COPY . .

# Expose React Native development server port
EXPOSE 19000

# Start the React Native app
CMD ["npm", "start"]
