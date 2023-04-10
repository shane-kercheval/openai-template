# project-template

This repo contains a minimal project template for making asynchronous OpenAI API calls. It has additional functionality that, for example, parses the responses and calculates the costs of API calls.

---

This project contains

- dockerfile
- linting
- unit tests
- doctests
- code coverage
- makefile and command line program (via click)

# Running the Project

All commmands for running the project can be found in the `Makefile` and ran in the command-line with the command `make [command]`.

## Starting Docker

Build and run docker-compose:

```commandline
make docker_run
```

```commandline
make docker_open
```

Running the entire project (tests, ETL, EDA, etc.) from command-line (outside of docker container):

```commandline
make docker_all
```

Running the entire project (tests, ETL, EDA, etc.) from command-line (inside docker container):

```commandline
make all
```

## Running the Code

The `Makefile` runs all components of the project. You can think of it as containing the implicit DAG, or recipe, of the project.

**Run make commands from terminal connected to container via `make docker_run` or `make zsh`**.

If you want to run the entire project from start to finish, including unit tests and linting, run:

```
make all
```
