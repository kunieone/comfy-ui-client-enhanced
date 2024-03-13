import axios, { AxiosRequestConfig } from 'axios';
import FormData from 'form-data';
import { writeFile } from 'fs/promises';
import { join } from 'path';

import pino from 'pino';
import WebSocket from 'ws';

import type {
  EditHistoryRequest,
  FolderName,
  HistoryResult,
  ImageContainer,
  ImageRef,
  ImagesResponse,
  ObjectInfoResponse,
  Prompt,
  PromptQueueResponse,
  QueuePromptResult,
  QueueResponse,
  ResponseError,
  SystemStatsResponse,
  UploadImageResult,
  ViewMetadataResponse,
} from './types.js';

// TODO: Make logger customizable
const logger = pino({
  level: 'info',
});

export class ComfyUIClient {
  public serverAddress: string;
  public clientId: string;
  public useHttps: boolean;
  public useWss: boolean;

  protected ws?: WebSocket;
  protected protocol: 'http' | 'https';
  protected wsProtocol: 'ws' | 'wss';

  constructor(serverAddress: string, clientId: string, useHttps = false, useWss = false) {
    this.serverAddress = serverAddress;
    this.clientId = clientId;
    this.useHttps = useHttps;
    this.useWss = useWss;
    this.protocol = useHttps ? 'https' : 'http';
    this.wsProtocol = useWss ? 'wss' : 'ws';
  }

  connect(timeout?: number, onTimeout?: () => void) {
    return new Promise<void>(async (resolve, reject) => {
      if (this.ws) {
        await this.disconnect();
      }

      const url = `${this.wsProtocol}://${this.serverAddress}/ws?clientId=${this.clientId}`;

      logger.info(`Connecting to url: ${url}`);

      this.ws = new WebSocket(url, {
        perMessageDeflate: false,
      });

      const timeoutId = timeout && setTimeout(() => {
        this.disconnect();
        onTimeout && onTimeout();
        reject(new Error('Connection timeout'));
      }, timeout);

      this.ws.on('open', () => {
        logger.info('Connection open');
        timeoutId && clearTimeout(timeoutId);
        resolve();
      });

      this.ws.on('close', () => {
        logger.info('Connection closed');
      });

      this.ws.on('error', (err) => {
        logger.error({ err }, 'WebSockets error');
        reject(err);
      });

      this.ws.on('message', (data, isBinary) => {
        if (isBinary) {
          logger.debug('Received binary data');
        } else {
          logger.debug('Received data: %s', data.toString());
        }
      });
    });
  }

  async disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = undefined;
    }
  }

  async getEmbeddings(): Promise<string[]> {
    const res = await axios.get(`${this.protocol}://${this.serverAddress}/embeddings`);
    return res.data;
  }

  async getExtensions(): Promise<string[]> {
    const res = await axios.get(`${this.protocol}://${this.serverAddress}/extensions`);
    return res.data;
  }

  async queuePrompt(prompt: Prompt): Promise<QueuePromptResult> {
    const res = await axios.post(`${this.protocol}://${this.serverAddress}/prompt`, {
      prompt,
      client_id: this.clientId,
    });
    return res.data;
  }

  async interrupt(): Promise<void> {
    const res = await axios.post(`${this.protocol}://${this.serverAddress}/interrupt`);
    return res.data;
  }

  async editHistory(params: EditHistoryRequest): Promise<void> {
    const res = await axios.post(`${this.protocol}://${this.serverAddress}/history`, params);
    return res.data;
  }

  async uploadImage(
    image: Buffer,
    filename: string,
    overwrite?: boolean,
  ): Promise<UploadImageResult> {
    const formData = new FormData();
    formData.append('image', new Blob([image]), filename);

    if (overwrite !== undefined) {
      formData.append('overwrite', overwrite.toString());
    }

    const res = await axios.post(`${this.protocol}://${this.serverAddress}/upload/image`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return res.data;
  }

  async uploadMask(
    image: Buffer,
    filename: string,
    originalRef: ImageRef,
    overwrite?: boolean,
  ): Promise<UploadImageResult> {
    const formData = new FormData();
    formData.append('image', new Blob([image]), filename);
    formData.append('originalRef', JSON.stringify(originalRef));

    if (overwrite !== undefined) {
      formData.append('overwrite', overwrite.toString());
    }

    const res = await axios.post(`${this.protocol}://${this.serverAddress}/upload/mask`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return res.data;
  }

  async getImage(
    filename: string,
    subfolder: string,
    type: string,
  ): Promise<Buffer> {
    const res = await axios.get(`${this.protocol}://${this.serverAddress}/view`, {
      params: {
        filename,
        subfolder,
        type,
      },
      responseType: 'arraybuffer',
    });
  
    return Buffer.from(res.data);
  }

  async viewMetadata(
    folderName: FolderName,
    filename: string,
  ): Promise<ViewMetadataResponse> {
    try {
      const res = await axios.get(
        `${this.protocol}://${this.serverAddress}/view_metadata/${folderName}?filename=${filename}`,
      );
      return res.data;
    } catch (err: any) {
      if (err.response && err.response.data) {
        throw new Error(JSON.stringify(err.response.data));
      } else {
        throw err;
      }
    }
  }

  async getSystemStats(): Promise<SystemStatsResponse> {
    const res = await axios.get(`${this.protocol}://${this.serverAddress}/system_stats`);
    return res.data;
  }

  async getPrompt(): Promise<PromptQueueResponse> {
    const res = await axios.get(`${this.protocol}://${this.serverAddress}/prompt`);
    return res.data;
  }

  async getObjectInfo(nodeClass?: string): Promise<ObjectInfoResponse> {
    const res = await axios.get(
      `${this.protocol}://${this.serverAddress}/object_info` +
        (nodeClass ? `/${nodeClass}` : ''),
    );
    return res.data;
  }

  async getHistory(promptId?: string): Promise<HistoryResult> {
    const res = await axios.get(
      `${this.protocol}://${this.serverAddress}/history` + (promptId ? `/${promptId}` : ''),
    );
    return res.data;
  }

  async getQueue(): Promise<QueueResponse> {
    const res = await axios.get(`${this.protocol}://${this.serverAddress}/queue`);
    return res.data;
  }

  async saveImages(response: ImagesResponse, outputDir: string) {
    for (const nodeId of Object.keys(response)) {
      for (const img of response[nodeId]) {
        const outputPath = join(outputDir, img.image.filename);
        await writeFile(outputPath, img.buffer);
      }
    }
  }

  
  async getImages(prompt: Prompt): Promise<ImagesResponse> {
    if (!this.ws) {
      throw new Error(
        'WebSocket client is not connected. Please call connect() before interacting.',
      );
    }
  
    const queue = await this.queuePrompt(prompt);
    const promptId = queue.prompt_id;
  
    return new Promise<ImagesResponse>((resolve, reject) => {
      const outputImages: ImagesResponse = {};
  
      const onMessage = async (data: WebSocket.RawData, isBinary: boolean) => {
        // Previews are binary data
        if (isBinary) {
          return;
        }
  
        try {
          const message = JSON.parse(data.toString());
          if (message.type === 'executing') {
            const messageData = message.data;
            if (!messageData.node) {
              const donePromptId = messageData.prompt_id;
  
              logger.info(`Done executing prompt (ID: ${donePromptId})`);
  
              // Execution is done
              if (messageData.prompt_id === promptId) {
                // Get history
                const historyRes = await this.getHistory(promptId);
                const history = historyRes[promptId];
  
                // Populate output images
                for (const nodeId of Object.keys(history.outputs)) {
                  const nodeOutput = history.outputs[nodeId];
                  if (nodeOutput.images) {
                    const imagesOutput: ImageContainer[] = [];
                    for (const image of nodeOutput.images) {
                      const buffer = await this.getImage(
                        image.filename,
                        image.subfolder,
                        image.type,
                      );
                      imagesOutput.push({
                        buffer,
                        image,
                      });
                    }
  
                    outputImages[nodeId] = imagesOutput;
                  }
                }
  
                // Remove listener
                this.ws?.off('message', onMessage);
                return resolve(outputImages);
              }
            }
          }
        } catch (err) {
          return reject(err);
        }
      };
  
      // Add listener
      this.ws?.on('message', onMessage);
    });
  }
}