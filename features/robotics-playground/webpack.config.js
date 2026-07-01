const { ModuleFederationPlugin } = require('@module-federation/enhanced/webpack');
const path = require('path');
const deps = require('./package.json').dependencies;

module.exports = {
  entry: './src/index',
  output: {
    path: path.resolve(__dirname, 'dist'),
    publicPath: 'auto',
    clean: true,
  },
  resolve: {
    extensions: ['.ts', '.tsx', '.js'],
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
  plugins: [
    new ModuleFederationPlugin({
      name: 'physicalAi',
      filename: 'remoteEntry.js',
      runtime: false,
      exposes: {
        './extensions': './src/extensions',
      },
      shared: {
        react: { singleton: true, requiredVersion: deps['react'] },
        'react-dom': { singleton: true, requiredVersion: deps['react-dom'] },
        'react-router-dom': { singleton: true, requiredVersion: deps['react-router-dom'] },
        '@patternfly/react-core': { singleton: true, requiredVersion: deps['@patternfly/react-core'] },
      },
    }),
  ],
};
