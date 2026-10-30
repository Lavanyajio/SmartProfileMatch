// Declarative pipeline: lint -> test -> docker build -> smoke test -> (optional) push -> (optional) deploy
// Jenkins needs: Docker on the agent, Python 3.10+, and these credentials (only for push/deploy):
//   'docker-registry-creds' (username/password)   'hf-token' (secret text)
pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '15'))
        disableConcurrentBuilds()
    }

    parameters {
        string(name: 'DOCKER_REGISTRY', defaultValue: '', description: 'e.g. docker.io/myuser — leave empty to skip push')
        booleanParam(name: 'DEPLOY', defaultValue: false, description: 'Run the container on this Jenkins host after a successful build')
        string(name: 'DEPLOY_PORT', defaultValue: '7861', description: 'Host port for the deployed app')
    }

    environment {
        APP_NAME   = 'hybrid-rag-assistant'
        IMAGE_TAG  = "${env.BUILD_NUMBER}"
        SMOKE_NAME = "${APP_NAME}-smoke-${env.BUILD_NUMBER}"
    }

    stages {
        stage('Checkout') {
            steps { checkout scm }
        }

        stage('Setup Python') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements-dev.txt
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '. .venv/bin/activate && flake8 .'
            }
        }

        stage('Unit Tests') {
            steps {
                sh '''
                    . .venv/bin/activate
                    mkdir -p reports
                    pytest --junitxml=reports/junit.xml --cov=. --cov-report=xml:reports/coverage.xml
                '''
            }
            post {
                always { junit allowEmptyResults: true, testResults: 'reports/junit.xml' }
            }
        }

        stage('Docker Build') {
            steps {
                sh 'docker build -t ${APP_NAME}:${IMAGE_TAG} -t ${APP_NAME}:latest .'
            }
        }

        stage('Smoke Test') {
            steps {
                sh '''
                    docker run -d --name ${SMOKE_NAME} -p 7860 ${APP_NAME}:${IMAGE_TAG}
                    PORT=$(docker port ${SMOKE_NAME} 7860/tcp | head -n1 | awk -F: '{print $NF}')
                    for i in $(seq 1 30); do
                        if curl -fsS "http://localhost:${PORT}/" > /dev/null; then
                            echo "App is up on port ${PORT}"; exit 0
                        fi
                        sleep 2
                    done
                    echo "App did not start in time"; docker logs ${SMOKE_NAME}; exit 1
                '''
            }
        }

        stage('Push Image') {
            when { expression { return params.DOCKER_REGISTRY?.trim() } }
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-registry-creds',
                                                  usernameVariable: 'REG_USER', passwordVariable: 'REG_PASS')]) {
                    sh '''
                        REGISTRY_HOST=$(echo "${DOCKER_REGISTRY}" | cut -d/ -f1)
                        echo "$REG_PASS" | docker login "$REGISTRY_HOST" -u "$REG_USER" --password-stdin
                        docker tag ${APP_NAME}:${IMAGE_TAG} ${DOCKER_REGISTRY}/${APP_NAME}:${IMAGE_TAG}
                        docker tag ${APP_NAME}:${IMAGE_TAG} ${DOCKER_REGISTRY}/${APP_NAME}:latest
                        docker push ${DOCKER_REGISTRY}/${APP_NAME}:${IMAGE_TAG}
                        docker push ${DOCKER_REGISTRY}/${APP_NAME}:latest
                    '''
                }
            }
        }

        stage('Deploy') {
            when { expression { return params.DEPLOY } }
            steps {
                withCredentials([string(credentialsId: 'hf-token', variable: 'HF_TOKEN')]) {
                    sh '''
                        docker rm -f ${APP_NAME} || true
                        docker volume create ${APP_NAME}-storage
                        docker run -d --name ${APP_NAME} --restart unless-stopped \
                            -p ${DEPLOY_PORT}:7860 -e HF_TOKEN \
                            -v ${APP_NAME}-storage:/app/storage \
                            ${APP_NAME}:${IMAGE_TAG}
                    '''
                }
            }
        }
    }

    post {
        always {
            sh 'docker rm -f ${SMOKE_NAME} || true'
            archiveArtifacts allowEmptyArchive: true, artifacts: 'reports/**'
        }
        success { echo "✅ ${APP_NAME}:${IMAGE_TAG} built and verified" }
        failure { echo "❌ Build failed — check the stage logs above" }
    }
}
