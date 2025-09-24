# --- STAGE 1: Build the Next.js application ---
FROM node:18-alpine AS builder

# Set the working directory
WORKDIR /app

# Copy package.json and lock files to leverage Docker caching
COPY package*.json ./

# Install all dependencies
RUN npm install

# Copy the rest of the application source code
COPY . .

# IMPORTANT: Set the API URL that the frontend will use during the build process
# This will be passed in from docker-compose.yml
ARG NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}

# Build the Next.js application for production
RUN npm run build

# --- STAGE 2: Create the final, small production image ---
FROM node:18-alpine

WORKDIR /app

# Copy only the necessary files from the 'builder' stage
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Expose the port Next.js runs on
EXPOSE 3000

# The command to start the production server
CMD ["npm", "start"]