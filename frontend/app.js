import ReactNative from 'react-native';
import HelloWorld from './testing.js';


const { AppRegistry } = ReactNative;


function startApp() {
    // Register your component with AppRegistry
    AppRegistry.registerComponent('HelloWorld', () => HelloWorld);

    AppRegistry.runApplication('HelloWorld', {
        rootTag: document.getElementById('root') || document.getElementById('app-root'),
    });
}

startApp();
