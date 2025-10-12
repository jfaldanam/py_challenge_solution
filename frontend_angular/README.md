# How the frontend is implemented?
The frontend is implemented using typescript and Angular to implement a web interface.

### Prerequisites
- Node.js and npm installed. You can download them from [nodejs.org](https://nodejs.org/en/download).
- Angular CLI installed globally. You can install it using npm: `npm install -g @angular/cli`.

# Deployment
First, the backend described at [../backend/](../backend/) must be deployed.

After deploying the backend, for instructions on how to deploy the frontend:

To start a local development server, run:

```bash
ng serve
```

Once the server is running, open your browser and navigate to `http://localhost:4200/`. The application will automatically reload whenever you modify any of the source files.

## Building

To build the project run:

```bash
ng build
```

This will compile your project and store the build artifacts in the `dist/` directory. By default, the production build optimizes your application for performance and speed.

## Deployment on Docker

To build the image run:
```bash
$ docker build -t py_challenge_frontend_angular:latest .
```

Then to run the API use:
```bash
$ docker run -v ./db:/data -p 4200:4200 py_challenge_frontend_angular:latest
```

Please note the use of a volume to persist the data.