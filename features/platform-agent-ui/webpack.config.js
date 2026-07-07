const { ModuleFederationPlugin } = require("@module-federation/enhanced/webpack");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const path = require("path");

module.exports = {
  entry: "./src/index",
  output: {
    path: path.resolve(__dirname, "dist"),
    publicPath: "auto",
    clean: true,
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"],
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        loader: "ts-loader",
        exclude: /node_modules/,
      },
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader"],
      },
    ],
  },
  devServer: {
    port: 9201,
    proxy: [
      {
        context: ["/api"],
        target: "http://localhost:8000",
      },
    ],
  },
  plugins: [
    new ModuleFederationPlugin({
      name: "platformAgent",
      filename: "remoteEntry.js",
      runtime: false,
      exposes: {
        "./extensions": "./src/extensions",
      },
      shared: {
        react: { singleton: true },
        "react-dom": { singleton: true },
        "react-router-dom": { singleton: true },
        "@patternfly/react-core": { singleton: true },
      },
    }),
    new HtmlWebpackPlugin({
      template: path.resolve(__dirname, "src/index.html"),
    }),
  ],
};
